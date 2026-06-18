"""Docker container sandbox for secure code execution.

Provides isolated execution environments for running linters, test suites,
and static analysis tools on untrusted code without risking the host system.
"""

from dataclasses import dataclass, field
import tempfile
from pathlib import Path
from typing import Any

import docker
from docker.errors import ContainerError, ImageNotFound, APIError
from docker.models.containers import Container

from code_review_agent.config import get_settings
from code_review_agent.logging import get_logger

logger = get_logger("docker_sandbox")


@dataclass
class SandboxResult:
    """Result from a sandboxed command execution.

    Attributes:
        exit_code: Process exit code (0 = success).
        stdout: Standard output content.
        stderr: Standard error content.
        timed_out: Whether execution hit the timeout.
        duration_seconds: How long execution took.
        command: The command that was run.
        image: Docker image used.
    """

    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_seconds: float = 0.0
    command: str = ""
    image: str = ""

    @property
    def success(self) -> bool:
        """Check if the command completed successfully."""
        return self.exit_code == 0 and not self.timed_out

    @property
    def output(self) -> str:
        """Combined stdout and stderr output."""
        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(self.stderr)
        return "\n".join(parts)


# Pre-built images for common language environments
SANDBOX_IMAGES: dict[str, str] = {
    "python": "python:3.11-slim",
    "node": "node:20-slim",
    "go": "golang:1.22-alpine",
    "rust": "rust:1.75-slim",
    "general": "ubuntu:22.04",
}


class DockerSandbox:
    """Manages Docker containers for isolated code execution.

    Creates short-lived containers with resource limits, mounts source
    code as read-only volumes, and captures output for analysis.

    Attributes:
        client: Docker SDK client.
        timeout: Maximum execution time in seconds.
        memory_limit: Container memory limit.
        cpu_limit: CPU quota as a float (1.0 = one full core).
    """

    def __init__(
        self,
        docker_client: docker.DockerClient | None = None,
        timeout: int | None = None,
        memory_limit: str | None = None,
        cpu_limit: float | None = None,
    ) -> None:
        """Initialize the Docker sandbox.

        Args:
            docker_client: Pre-configured Docker client. Creates from env if None.
            timeout: Execution timeout in seconds.
            memory_limit: Memory limit string (e.g., "512m", "1g").
            cpu_limit: CPU limit as float (1.0 = one core).
        """
        settings = get_settings()
        self.timeout = timeout or settings.sandbox_timeout
        self.memory_limit = memory_limit or settings.sandbox_memory_limit
        self.cpu_limit = cpu_limit or settings.sandbox_cpu_limit

        if docker_client:
            self.client = docker_client
        else:
            try:
                self.client = docker.from_env()
                self.client.ping()
                logger.info("docker sandbox connected")
            except Exception as e:
                logger.error("failed to connect to Docker", error=str(e))
                raise

    def run_command(
        self,
        command: str,
        image: str = "python:3.11-slim",
        source_dir: str | Path | None = None,
        working_dir: str = "/workspace",
        environment: dict[str, str] | None = None,
        network_disabled: bool = True,
    ) -> SandboxResult:
        """Run a command in an isolated Docker container.

        Args:
            command: Shell command to execute.
            image: Docker image to use.
            source_dir: Host directory to mount as /workspace.
            working_dir: Working directory inside the container.
            environment: Environment variables for the container.
            network_disabled: Whether to disable network access.

        Returns:
            SandboxResult with output and exit code.
        """
        import time

        start_time = time.time()

        # Prepare volume mounts
        volumes: dict[str, dict[str, str]] = {}
        if source_dir:
            source_path = str(Path(source_dir).resolve())
            volumes[source_path] = {"bind": working_dir, "mode": "ro"}

        # Resource limits
        nano_cpus = int(self.cpu_limit * 1e9)

        try:
            # Ensure image exists
            self._ensure_image(image)

            # Run container
            container: Container = self.client.containers.run(
                image=image,
                command=["sh", "-c", command],
                working_dir=working_dir,
                volumes=volumes or None,
                environment=environment or {},
                network_disabled=network_disabled,
                mem_limit=self.memory_limit,
                nano_cpus=nano_cpus,
                detach=True,
                remove=False,
                stdout=True,
                stderr=True,
                read_only=True,
                tmpfs={"/tmp": "size=100M"},
            )

            # Wait for completion with timeout
            result = container.wait(timeout=self.timeout)
            exit_code = result.get("StatusCode", -1)

            # Capture output
            stdout = container.logs(stdout=True, stderr=False).decode(
                "utf-8", errors="replace"
            )
            stderr = container.logs(stdout=False, stderr=True).decode(
                "utf-8", errors="replace"
            )

            duration = time.time() - start_time

            # Cleanup container
            container.remove(force=True)

            return SandboxResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                timed_out=False,
                duration_seconds=round(duration, 2),
                command=command,
                image=image,
            )

        except docker.errors.ContainerError as e:
            duration = time.time() - start_time
            return SandboxResult(
                exit_code=e.exit_status,
                stdout="",
                stderr=str(e),
                timed_out=False,
                duration_seconds=round(duration, 2),
                command=command,
                image=image,
            )

        except Exception as e:
            duration = time.time() - start_time
            is_timeout = "timed out" in str(e).lower() or "read timeout" in str(e).lower()

            logger.warning(
                "sandbox execution error",
                command=command[:100],
                error=str(e),
                timed_out=is_timeout,
            )

            # Try to cleanup any orphan container
            try:
                if "container" in locals():
                    container.remove(force=True)
            except Exception:
                pass

            return SandboxResult(
                exit_code=-1,
                stdout="",
                stderr=str(e),
                timed_out=is_timeout,
                duration_seconds=round(duration, 2),
                command=command,
                image=image,
            )

    def run_with_source(
        self,
        command: str,
        source_files: dict[str, str],
        image: str = "python:3.11-slim",
        setup_command: str | None = None,
        environment: dict[str, str] | None = None,
    ) -> SandboxResult:
        """Run a command with source files written to a temp directory.

        Creates a temporary directory, writes all source files, mounts
        it into the container, and executes the command.

        Args:
            command: Command to execute.
            source_files: Mapping of relative_path -> file_content.
            image: Docker image.
            setup_command: Optional setup command to run before the main command.
            environment: Environment variables.

        Returns:
            SandboxResult with execution output.
        """
        with tempfile.TemporaryDirectory(prefix="cra_sandbox_") as tmpdir:
            tmp_path = Path(tmpdir)

            # Write source files
            for rel_path, content in source_files.items():
                file_path = tmp_path / rel_path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")

            # Combine setup and main command
            full_command = command
            if setup_command:
                full_command = f"{setup_command} && {command}"

            return self.run_command(
                command=full_command,
                image=image,
                source_dir=tmpdir,
                environment=environment,
                network_disabled=False if setup_command else True,
            )

    def _ensure_image(self, image: str) -> None:
        """Pull a Docker image if not already available locally.

        Args:
            image: Image name with optional tag.
        """
        try:
            self.client.images.get(image)
        except ImageNotFound:
            logger.info("pulling docker image", image=image)
            self.client.images.pull(image)

    def is_available(self) -> bool:
        """Check if Docker is available and responsive.

        Returns:
            True if Docker daemon is reachable.
        """
        try:
            self.client.ping()
            return True
        except Exception:
            return False
