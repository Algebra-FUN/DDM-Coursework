import re

with open('./blocklist.xml','r',encoding='utf-8') as f:
    content = f.read()
    matches = re.findall(r'<emItem blockID=".+" id="[^/\^\\]+@[^/\^\\]+\.(?:org|com)">', content)
    # actually you don't need to skip those "regex special characters \, /, ˆ" by you self,
    # since those regex won't end with "org" or "com", they end with "$/",
    # but I still add "\, /, ˆ" skipping pattern.
    # "[^/\^\\]", it looks like a emoji, funny.
    print(f'found {len(matches)} matches:')
    print(*matches,sep='\n')
