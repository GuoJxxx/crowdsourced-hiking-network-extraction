"""Bind transient desktop state to this computer and Windows account."""
import hashlib
import os


def host_id():
    machine = ''
    if os.name == 'nt':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\Microsoft\Cryptography',
                            access=winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as key:
            machine = str(winreg.QueryValueEx(key, 'MachineGuid')[0])
    identity = '|'.join((os.environ.get('COMPUTERNAME', ''),
                         os.environ.get('USERNAME', ''), machine)).lower()
    return hashlib.sha256(identity.encode('utf-8')).hexdigest()
