"""Create an independent managed-serving bundle from a trained component bank."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
from ..train import file_sha
from .managed_serving_bundle_v1 import clone_serving_bundle


def export_bank(bank_file, destination):
    bank_file=Path(bank_file).resolve(); root=bank_file.parent
    bank=json.loads(bank_file.read_text()); destination=Path(destination).absolute()
    if destination.exists(): raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='recokt-export-',dir=destination.parent) as tmp:
        source=Path(tmp)
        manifest={'status':'complete','architecture':bank['architecture'],
                  'initially_withdrawn':[],'components':{}}
        for name,info in bank['components'].items():
            if Path(name).name != name or name in ('','.','..'): raise ValueError('unsafe component name')
            original=(root/info['file']).resolve()
            if not original.is_relative_to(root) or file_sha(original)!=info['sha256']:
                raise ValueError('bank component path or hash mismatch')
            filename=name+'.pt'; shutil.copyfile(original,source/filename)
            manifest['components'][name]={'file':filename,'sha256':info['sha256'],
                                          'members':info['members']}
        (source/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        return clone_serving_bundle(source,destination)


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(export_bank(a.bank,a.output))

if __name__=='__main__':main()
