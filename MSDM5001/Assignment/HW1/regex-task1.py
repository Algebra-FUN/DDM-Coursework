import re

with open('./blocklist.xml','r',encoding='utf-8') as f:
    content = f.read()
    matches = re.findall(r'<emItem blockID="[id].*\d" id=".+">', content)
    print(f'found {len(matches)} matches:')
    print(*matches,sep='\n')
