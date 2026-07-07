#!/bin/sh
# Auto-grant the `app` user access to any passed-through GPU device nodes, then
# drop to that user. The device nodes' owning group GID varies per host, so we
# read it at runtime instead of hardcoding it in compose. On a CPU-only container
# (no device nodes) this is just the privilege drop. Everything is best-effort:
# a failure in the group setup must never stop the server from starting.
if [ "$(id -u)" = "0" ]; then
  for dev in /dev/kfd /dev/dri/renderD* /dev/dri/card* /dev/nvidia*; do
    [ -e "$dev" ] || continue
    gid="$(stat -c '%g' "$dev" 2>/dev/null)"
    [ -n "$gid" ] || continue
    [ "$gid" = "0" ] && continue
    grp="$(getent group "$gid" | cut -d: -f1)"
    if [ -z "$grp" ]; then grp="gpu$gid"; groupadd -g "$gid" "$grp" 2>/dev/null || true; fi
    usermod -aG "$grp" app 2>/dev/null || true
  done
  if command -v setpriv >/dev/null 2>&1; then
    exec setpriv --reuid "$(id -u app)" --regid "$(id -g app)" --init-groups "$@"
  fi
  exec "$@"   # no setpriv -> run as root (works; less isolated) rather than fail to boot
fi
exec "$@"
