>>> from anyscale import AnyscaleSDK as AS
>>> sdk = AS()
>>> import dotenv
>>> dotenv.load_dotenv()
True
>>> import os
>>> env = dict(os.environ)
>>> ANYSCALE_KEY = env['ANYSCALE_KEY']
>>> env.keys()
dict_keys(['SHELL', 'TERM_PROGRAM_VERSION', 'LLAMA', 'TMUX', 'MESA_DIR', 'PUBLIC_DIR', 'COMMUNITY_DIR', 'PWD', 'LOGNAME', 'XDG_SESSION_TYPE', 'PNPM_HOME', 'MOTD_SHOWN', 'HOME', 'LANG', 'LS_COLORS', 'MODEL', 'VIRTUAL_ENV', 'SSH_CONNECTION', 'LIBREWOLF_USER_DIR', 'NVM_DIR', 'TANGIBLE_DIR', 'LESSCLOSE', 'XDG_SESSION_CLASS', 'TERM', 'LESSOPEN', 'PREPEND_HELP', 'USER', 'MESA_PYTHON_DIR', 'TMUX_PANE', 'SUBLIME_USER_DIR', 'LIBREWOLF_FLATPACK_USER_DIR', 'SHLVL', 'XDG_SESSION_ID', 'VIRTUAL_ENV_PROMPT', 'XDG_RUNTIME_DIR', 'PS1', 'SSH_CLIENT', 'XDG_DATA_DIRS', 'PATH', 'HOMEBIN', 'DBUS_SESSION_BUS_ADDRESS', 'SSH_TTY', 'OLDPWD', 'TERM_PROGRAM', '_', 'DEBUG', 'LOGLEVEL', 'OPEN_AI_KEY', 'OPEN_ROUTER_KEY', 'TOGETHER_AI_KEY'])
>>> ls .env
>>> !nano .env
>>> dotenv.load_dotenv()
True
>>> env = dict(os.environ)
>>> ANYSCALE_KEY = env['ANYSCALE_KEY']
>>> sdk = AS(ANYSCALE_KEY)
>>> sdk
<anyscale.sdk.anyscale_client.sdk.AnyscaleSDK at 0x7f1a83498430>
>>> git clone git@github.com:CoTErrorRecovery/CoTErrorRecovery
>>> cd ..
>>> !git clone git@github.com:CoTErrorRecovery/CoTErrorRecovery
>>> from datasets import load_dataset
... ds = load_dataset("openai/gsm8k", "main")
...
>>> ds['train'][0]
{'question': 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?',
 'answer': 'Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72'}
>>> ds_socratic['train'][0]
>>> from datasets import load_dataset
... 
... ds_socratic = load_dataset("openai/gsm8k", 'socratic')
...
>>> ds_socratic['train'][0]
{'question': 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?',
 'answer': 'How many clips did Natalia sell in May? ** Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nHow many clips did Natalia sell altogether in April and May? ** Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72'}
>>> len(ds)
2
>>> len(ds['train'])
7473
>>> len(ds['test'])
1319
>>> hist -o -p -f README-gsm8k.md
