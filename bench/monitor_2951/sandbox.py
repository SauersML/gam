"""Run untrusted Python (an agent's func.py and test.py) with no network and no access outside one directory.

Layers, outermost first:
  - kernel: on Linux a new network namespace (unshare -rn: no interface but a downed loopback), on macOS
    sandbox-exec with a profile that denies all network and all file access outside the run directory
    and the interpreter's own files;
  - identity: dropped to nobody when the harness runs as root (the pod), so files private to root stay closed;
  - limits: CPU seconds, address space, file size, process count, and a wall-clock timeout that kills the
    process group;
  - interpreter: an audit hook installed before any agent code runs, which refuses every socket operation,
    every subprocess/exec/fork/spawn, ctypes, and every open/listdir/remove/rename/mkdir/chdir outside the run
    directory except reads of the interpreter's own installation.
selftest() runs known escapes through the same path and returns those that were not refused.
"""
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile

# The interpreter agent code runs under: the system's on Linux (a venv's may live under root's home, closed to
# nobody), the real binary behind a venv on macOS (sandbox-exec closes /Users). Only the standard library is needed.
PY = os.path.realpath(sys.executable)
if sys.platform.startswith("linux"):
    PY = next((p for p in ("/usr/bin/python3", "/usr/local/bin/python3") if os.path.exists(p)), PY)

# The audit hook, run with -c before the agent's script. ROOT is the run directory; READ are the prefixes the
# interpreter must read to import the standard library.
HOOK = r'''
import sys, os
ROOT = os.path.realpath(sys.argv[1]); SCRIPT = sys.argv[2]
READ = tuple(sorted({os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix)}))
_real = os.path.realpath
def _inside(p, roots):
    try:
        p = _real(os.fsdecode(p) if not isinstance(p, int) else "/proc/self/fd")
    except Exception:
        return False
    return any(p == r or p.startswith(r + os.sep) for r in roots)
DENY = ("socket.", "subprocess.", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty",
        "pty.", "ctypes.", "os.kill", "os.killpg", "signal.pthread_kill", "winreg.", "webbrowser.", "urllib.Request",
        "http.client.", "ftplib.", "smtplib.", "telnetlib.", "imaplib.", "poplib.", "nntplib.", "os.chroot", "os.setuid")
PATHS = {"os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.chmod", "os.chown", "os.link", "os.symlink",
         "os.truncate", "os.utime", "os.chdir", "os.chflags", "os.lchflags", "shutil.copyfile", "shutil.copymode",
         "shutil.copystat", "shutil.copytree", "shutil.move", "shutil.rmtree", "shutil.make_archive",
         "shutil.unpack_archive", "os.mkfifo", "os.mknod", "os.setxattr", "os.removexattr", "glob.glob",
         "os.scandir", "os.listdir", "os.walk", "os.fwalk"}
BLOCKED_MODULES = {"ctypes", "_ctypes", "_socket", "socket", "_posixsubprocess", "subprocess", "_ssl", "ssl",
                   "multiprocessing", "_multiprocessing", "pty", "fcntl", "mmap", "resource", "_testcapi", "_xxsubinterpreters",
                   "_interpreters", "asyncio", "selectors", "select", "urllib.request", "http.client", "requests"}
def hook(event, args):
    if event == "import":
        name = args[0] or ""
        if name in BLOCKED_MODULES or name.split(".")[0] in ("ctypes", "_ctypes"):
            raise PermissionError("sandbox: import of %s refused" % name)
        return
    if event == "open":
        path, mode = args[0], args[1]
        if path is None:
            return
        writing = isinstance(mode, str) and any(c in mode for c in "wax+")
        if isinstance(mode, int):
            writing = bool(mode & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        if _inside(path, (ROOT,)) or (not writing and _inside(path, READ)):
            return
        raise PermissionError("sandbox: open of %r refused" % (path,))
    if event.startswith(DENY):
        raise PermissionError("sandbox: %s refused" % event)
    if event in PATHS:
        for a in args[:2]:
            if isinstance(a, (str, bytes, os.PathLike)) and not (_inside(a, (ROOT,)) or (event in ("os.listdir", "os.scandir") and _inside(a, READ))):
                raise PermissionError("sandbox: %s of %r refused" % (event, a))
sys.addaudithook(hook)
del hook
class _Refuse:
    # importlib.import_module raises no "import" audit event, so blocked modules are also refused by a finder
    @staticmethod
    def find_spec(name, path=None, target=None):
        if name in BLOCKED_MODULES or name.split(".")[0] in ("ctypes", "_ctypes"):
            raise PermissionError("sandbox: import of %s refused" % name)
        return None
sys.meta_path.insert(0, _Refuse)
for _m in list(sys.modules):
    if _m in BLOCKED_MODULES:
        del sys.modules[_m]
sys.argv = [SCRIPT]
sys.path.insert(0, ROOT)
import runpy
runpy.run_path(SCRIPT, run_name="__main__")
'''

# macOS kernel layer: no network, file reads only of the run directory and the system and interpreter, writes only
# in the run directory.
SB_PROFILE = '''(version 1)
(allow default)
(deny network*)
(deny file-write*)
(deny file-read* (subpath "/Users") (subpath "/private/tmp") (subpath "/private/var/folders") (subpath "/Volumes") (subpath "/private/etc"))
(allow file-read* (subpath "{pyroot}") (subpath "{root}"))
(allow file-write* (subpath "{root}") (literal "/dev/null"))
'''


def _drop():
    # As root (the pod), become nobody before anything else: files private to root stay closed.
    if os.getuid() == 0:
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)


def _unshare_ok():
    # A user namespace (mapping nobody to its root) holding a new network namespace, entered after the drop.
    if not sys.platform.startswith("linux") or not shutil.which("unshare"):
        return False
    try:
        r = subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=10, preexec_fn=_drop)
        return r.returncode == 0
    except Exception:
        return False


UNSHARE = _unshare_ok()


def _limits(cpu, mem_gb):
    def f():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        if sys.platform.startswith("linux"):
            resource.setrlimit(resource.RLIMIT_AS, (int(mem_gb * 2**30), int(mem_gb * 2**30)))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 2**20, 64 * 2**20))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        _drop()
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        except Exception:
            pass
    return f


def run(root, script, timeout=30, cpu=20, mem_gb=4):
    """Run ROOT/SCRIPT under every layer; returns (exit code, stdout, stderr, timed out)."""
    root = os.path.realpath(root)
    if os.getuid() == 0:
        os.chmod(root, 0o777)
        for n in os.listdir(root):
            os.chmod(os.path.join(root, n), 0o666)
    pyroot = os.path.realpath(sys.base_prefix)
    cmd = [PY, "-E", "-s", "-B", "-c", HOOK, root, os.path.join(root, script)]
    if sys.platform == "darwin":
        prof = SB_PROFILE.format(root=root, pyroot=pyroot)
        extra = {os.path.realpath(sys.prefix), os.path.realpath(os.path.dirname(os.path.realpath(PY)))} - {pyroot}
        for p in extra:
            prof = prof.replace('(subpath "%s")' % root, '(subpath "%s") (subpath "%s")' % (root, p), 1)
        cmd = ["sandbox-exec", "-p", prof] + cmd
    elif UNSHARE:
        cmd = ["unshare", "-rn"] + cmd
    env = {"PATH": "/usr/bin:/bin", "HOME": root, "TMPDIR": root, "LANG": "C.UTF-8", "PYTHONHASHSEED": "0"}
    try:
        p = subprocess.Popen(cmd, cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, preexec_fn=_limits(cpu, mem_gb))
        try:
            out, err = p.communicate(timeout=timeout)
            return p.returncode, out.decode("utf-8", "replace"), err.decode("utf-8", "replace"), False
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, 9)
            except Exception:
                p.kill()
            out, err = p.communicate()
            return -9, out.decode("utf-8", "replace"), err.decode("utf-8", "replace"), True
    except Exception as e:
        return -1, "", "sandbox failure: %r" % (e,), False


def fresh(files):
    """A new run directory holding FILES ({name: text})."""
    d = tempfile.mkdtemp(prefix="sbx_")
    for n, t in files.items():
        with open(os.path.join(d, n), "w") as f:
            f.write(t)
    return d


ESCAPES = {
    "tcp connect": "import socket; s=socket.create_connection(('1.1.1.1', 53), timeout=3); print('ESCAPED')",
    "raw _socket": "s=__import__('_socket').socket(); s.connect(('8.8.8.8', 53)); print('ESCAPED')",
    "https fetch": "import urllib.request; urllib.request.urlopen('https://huggingface.co', timeout=5); print('ESCAPED')",
    "dns lookup": "import socket; socket.getaddrinfo('huggingface.co', 443); print('ESCAPED')",
    "read /etc/passwd": "print('ESCAPED' if open('/etc/passwd').read() else '')",
    "read harness": "print('ESCAPED' if open(%r).read() else '')" % os.path.realpath(__file__),
    "read parent dir file": "import os; print('ESCAPED' if open(os.path.join('..', os.listdir('..')[0])).read(1) is not None else '')",
    "list /": "import os; os.listdir('/'); print('ESCAPED')",
    "write outside": "open('/tmp/sbx_escape_probe', 'w').write('x'); print('ESCAPED')",
    "write home": "open(%r, 'w').write('x'); print('ESCAPED')" % os.path.expanduser("~/sbx_escape_probe"),
    "subprocess curl": "import subprocess; subprocess.run(['curl', '-s', 'https://example.com']); print('ESCAPED')",
    "os.system": "import os; os.system('touch /tmp/sbx_escape_probe2'); print('ESCAPED')",
    "os.fork": "import os; pid=os.fork(); print('ESCAPED')",
    "ctypes": "import ctypes; ctypes.CDLL(None); print('ESCAPED')",
    "importlib ctypes": "import importlib; importlib.import_module('_ctypes').dlopen(None); print('ESCAPED')",
    "meta_path removed": "import sys, importlib; sys.meta_path.pop(0); importlib.import_module('_socket').socket().connect(('1.1.1.1', 53)); print('ESCAPED')",
    "builtins open via loader": "import io, _io; _io.open('/etc/passwd').read(); print('ESCAPED')",
    "posix spawn": "import os; os.posix_spawn('/bin/sh', ['sh', '-c', 'true'], {}); print('ESCAPED')",
    "exec via posix": "import posix; posix.execv('/bin/sh', ['sh']); print('ESCAPED')",
    "remove outside": "import os; os.remove('/etc/hostname'); print('ESCAPED')",
    "chdir out": "import os; os.chdir('/'); print('ESCAPED')",
    "io.open outside": "import io; io.FileIO('/etc/passwd').read(); print('ESCAPED')",
    "os.open outside": "import os; os.read(os.open('/etc/passwd', os.O_RDONLY), 10); print('ESCAPED')",
}


def selftest():
    """Every known escape must fail, the run directory must stay usable, and a busy loop must be killed."""
    leaks = []
    for name, code in ESCAPES.items():
        d = fresh({"t.py": "print('STARTED', flush=True)\n" + code + "\n"})
        rc, out, err, _ = run(d, "t.py", timeout=20)
        if "ESCAPED" in out or "STARTED" not in out:
            leaks.append(name if "STARTED" in out else name + " (interpreter did not start: %s)" % err[-200:])
        shutil.rmtree(d, ignore_errors=True)
    for p in ("/tmp/sbx_escape_probe", "/tmp/sbx_escape_probe2", os.path.expanduser("~/sbx_escape_probe")):
        if os.path.exists(p):
            leaks.append("file appeared: " + p)
            os.remove(p)
    d = fresh({"func.py": "def f():\n    return 41\n", "t.py": "from func import f\nopen('w.txt','w').write('ok')\nimport json, math, collections, itertools, functools, heapq, bisect\nassert f() == 41\nprint('INSIDE OK', open('w.txt').read())\n"})
    rc, out, err, _ = run(d, "t.py")
    inside_ok = "INSIDE OK ok" in out
    shutil.rmtree(d, ignore_errors=True)
    d = fresh({"t.py": "while True: pass\n"})
    rc, out, err, timed_out = run(d, "t.py", timeout=4, cpu=2)
    killed = rc != 0
    shutil.rmtree(d, ignore_errors=True)
    return {"leaks": leaks, "inside_ok": inside_ok, "loop_killed": killed, "kernel": "sandbox-exec" if sys.platform == "darwin" else ("unshare -rn (network namespace)" if UNSHARE else "none: uid nobody + audit hook only"), "uid_drop": os.getuid() == 0, "escapes_tried": len(ESCAPES)}


if __name__ == "__main__":
    r = selftest()
    print(json.dumps(r))
    sys.exit(0 if not r["leaks"] and r["inside_ok"] and r["loop_killed"] else 1)
