import re
from pathlib import Path

from tests.test_workflows import Step, run_step

ROOT = Path(__file__).resolve().parents[1]


def test_gpu_setup_returns_to_the_repository_before_the_project_command(tmp_path: Path):
    text = (ROOT / "docs/reproducibility.md").read_text()
    section = text.split("## Small GPU through faster-weathernext")[1]
    blocks = re.findall(r"```sh\n(.*?)\n```", section, re.DOTALL)
    (tmp_path / "envs/fwn").mkdir(parents=True)
    script = """
set -e
uv() {
  if [ "$1" = "sync" ] && [ "${2:-}" = "--project" ]; then
    test -d "$3" || return 2
  fi
}
"""
    script += blocks[0] + "\n" + blocks[1].replace("<id>", "example") + "\npwd\n"
    result = run_step(Step(run=script), tmp_path, tmp_path)
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == tmp_path.resolve()
