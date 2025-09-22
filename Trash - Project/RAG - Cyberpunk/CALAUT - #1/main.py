import requests

r = requests.get('https://aurion.edhec.edu/faces/Planning.xhtml#', auth=('user', 'pass'))
print(r.status_code, r.headers, r.encoding)

#aborted no need for anymore / configured with account merging