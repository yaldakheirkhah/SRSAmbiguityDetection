"""Create this experiment's isolated environment only when storage is sufficient."""
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def main():
    root=Path(__file__).resolve().parents[1]
    free=shutil.disk_usage(root).free
    report={"timestamp_utc":datetime.now(timezone.utc).isoformat(),
            "status":"BLOCKED_ENVIRONMENT_NOT_CREATED", "free_disk_bytes":free,
            "minimum_before_install_gib":16, "minimum_after_install_gib":12,
            "training_executed":False, "held_out_data_accessed":False, "reasons":[]}
    if platform.system()!='Darwin' or platform.machine()!='arm64':
        report['reasons'].append('Native arm64 macOS is required.')
    if sys.version_info[:2]!=(3,12): report['reasons'].append('Use native Python 3.12.')
    environment=root/'.venv'
    if free < 16*1024**3:
        report['reasons'].append('At least 16GiB actually free is required before installing the isolated environment; 20GiB or more is recommended.')
    if environment.exists():
        report['reasons'].append('Environment already exists; preserve it and use the notebook/preflight to verify it. Setup never overwrites an environment.')
    if not report['reasons']:
        try:
            from verify_inputs import verify_inputs
            verify_inputs(root)
            subprocess.run([sys.executable,'-m','venv',str(environment)],check=True)
            python=environment/'bin/python'
            subprocess.run([str(python),'-m','pip','install','--disable-pip-version-check','--no-cache-dir',
                            '-r',str(root/'requirements_mac.txt')],check=True)
            report['free_disk_after_install_bytes']=shutil.disk_usage(root).free
            if report['free_disk_after_install_bytes'] < 12*1024**3:
                report['reasons'].append('Installation finished but less than12GiB remains. Training is blocked; preserve the environment.')
                report['status']='BLOCKED_AFTER_ENVIRONMENT_INSTALL'
            else:
                report['status']='ENVIRONMENT_CREATED_TRAINING_NOT_STARTED'
        except Exception as exc:
            report['status']='ENVIRONMENT_SETUP_FAILED_PRESERVE_PARTIAL'
            report['reasons'].append(repr(exc))
    (root/'reports').mkdir(exist_ok=True)
    (root/'reports/ENVIRONMENT_SETUP_STATUS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if report['status']=='ENVIRONMENT_CREATED_TRAINING_NOT_STARTED' else 2


if __name__=='__main__': raise SystemExit(main())
