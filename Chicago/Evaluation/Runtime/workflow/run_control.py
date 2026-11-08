"""Pause and resume the process tree belonging to one experiment worker."""
from __future__ import annotations

from pathlib import Path
import time

import psutil


def recorded_process(record):
    """A reused PID is a different process, even when its program name matches."""
    try:
        process = psutil.Process(int(record['pid']))
        if abs(process.create_time() - float(record['created'])) > .01:
            return None
        return process if process.is_running() else None
    except (psutil.NoSuchProcess, KeyError, TypeError, ValueError):
        return None


def _worker(state, root, *, allow_check=False):
    allowed = ('run', 'check') if allow_check else ('run',)
    if state.get('action') not in allowed:
        raise RuntimeError('This control does not apply to the recorded task.')
    try:
        process = psutil.Process(int(state['pid']))
        expected = state.get('process_created', state.get('started'))
        tolerance = .01 if state.get('process_created') else 60
        if expected is None or abs(process.create_time() - float(expected)) > tolerance:
            raise RuntimeError('The recorded worker identity has changed.')
        arguments = process.cmdline()
        module = arguments.index('-m')
        if arguments[module+1:module+3] != ['workflow.engine', 'worker']:
            raise ValueError('worker command')
        action = arguments.index('--action', module+3)
        if arguments[action+1] != state['action'] or Path(process.cwd()).resolve() != Path(root).resolve():
            raise ValueError('worker location')
        return process
    except (psutil.NoSuchProcess, KeyError, TypeError, ValueError, IndexError) as error:
        raise RuntimeError('The recorded experiment worker is unavailable or its identity changed.') from error


def suspend_run(state, root, on_change=None, *, allow_check=False):
    """Freeze parents first so that no next task starts while children pause."""
    worker = _worker(state, root, allow_check=allow_check)
    paused = {'started': time.time(), 'processes': []}
    seen = set()
    pending = [worker]
    try:
        for _ in range(32):
            for process in pending:
                try:
                    record = {'pid': process.pid, 'created': process.create_time(), 'suspended': False}
                    identity = (record['pid'], record['created'])
                    if identity in seen:
                        continue
                    process.suspend()
                    record['suspended'] = True
                    paused['processes'].append(record)
                    seen.add(identity)
                    if on_change:
                        on_change(paused)
                except psutil.NoSuchProcess:
                    continue
            pending = []
            for process in worker.children(recursive=True):
                try:
                    if (process.pid, process.create_time()) not in seen:
                        pending.append(process)
                except psutil.NoSuchProcess:
                    continue
            if not pending:
                return paused
        raise RuntimeError('The experiment process tree is still changing. Try Pause again.')
    except Exception:
        # Unwind only suspension operations performed by this request.
        for record in reversed(paused['processes']):
            process = recorded_process(record)
            if process and record['suspended']:
                try:
                    process.resume()
                    record['suspended'] = False
                except psutil.Error:
                    pass
        if on_change:
            on_change(paused)
        raise


def resume_run(state, root, paused, on_change=None):
    """Resume saved children first, then the same coordinator process."""
    worker = _worker(state, root)
    for record in reversed(paused['processes']):
        if not record.get('suspended'):
            continue
        process = recorded_process(record)
        if process is None:
            continue
        if process.pid == worker.pid:
            # Persist this transition before releasing the worker, which can
            # immediately publish a newer stage or completion status.
            record['suspended'] = False
            if on_change:
                on_change(paused)
            try:
                process.resume()
            except Exception:
                record['suspended'] = True
                if on_change:
                    on_change(paused)
                raise
        else:
            process.resume()
            record['suspended'] = False
            if on_change:
                on_change(paused)


def has_suspended_processes(paused):
    return any(record.get('suspended') and recorded_process(record) is not None
               for record in paused.get('processes', []))


def stop_run(state, root, on_change=None):
    """Freeze creation of descendants, then terminate children before their owner."""
    worker = _worker(state, root, allow_check=True)
    paused = state.get('paused') or suspend_run(state, root, on_change, allow_check=True)
    children = []
    for record in reversed(paused['processes']):
        process = recorded_process(record)
        if process is not None and process.pid != worker.pid:
            process.kill()
            children.append(process)
    _, alive = psutil.wait_procs(children, timeout=5)
    if alive:
        raise RuntimeError('Some experiment children could not be stopped; the coordinator remains paused.')
    worker.kill()
    worker.wait(timeout=5)
