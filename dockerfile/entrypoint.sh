#!/bin/sh

# Add user to /etc/passwd if it doesn't exist
if ! id -u $USER_ID >/dev/null 2>&1; then
    echo "myuser:x:$USER_ID:$GROUP_ID::/home/myuser:/bin/sh" >> /etc/passwd
    echo "myuser:x:$GROUP_ID:" >> /etc/group
fi

exec gosu myuser "$@"
