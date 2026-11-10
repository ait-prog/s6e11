import json, os, sys, subprocess

R = 'd:/hack/s6e11'
KG = 'D:/hack/tools-env/Scripts/kaggle.exe'


def mk(n, src, J=None, gpu=True, ds=(), kn=(), push=True):
    d = f'{R}/k/dev{n}'
    os.makedirs(d, exist_ok=True)
    s = open(f'{R}/src/{src}', encoding='utf-8').read()
    if J is not None:
        s = s.replace('\nJ = []\n', f'\nJ = {J!r}\n', 1)
    open(f'{d}/run.py', 'w', encoding='utf-8').write(s)
    m = dict(id=f'aitbekovalibek/s6e11-dev{n}', title=f's6e11-dev{n}', code_file='run.py', language='python',
             kernel_type='script', is_private=True, enable_gpu=gpu, enable_tpu=False, enable_internet=gpu,
             dataset_sources=list(ds), competition_sources=['playground-series-s6e11'], kernel_sources=list(kn))
    json.dump(m, open(f'{d}/kernel-metadata.json', 'w'), indent=1)
    if push:
        a = [KG, 'kernels', 'push', '-p', d] + (['--accelerator', 'NvidiaTeslaT4'] if gpu else [])
        print(subprocess.run(a, capture_output=True, text=True).stdout.strip().splitlines()[-1])


def get(n, sub=''):
    d = f'{R}/out/dev{n}'
    os.makedirs(d, exist_ok=True)
    r = subprocess.run([KG, 'kernels', 'output', f'aitbekovalibek/s6e11-dev{n}', '-p', d, '-o'], capture_output=True, text=True)
    print(r.stdout.strip()[-300:], r.stderr.strip()[-300:])
    return d


if __name__ == '__main__':
    get(int(sys.argv[1]))
