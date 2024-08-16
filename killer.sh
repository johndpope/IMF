#!/usr/bin/env bash

# Author: Oleh Pshenychnyi
# Date: 13.02.2021
#
# Kill all processes matching a provided pattern.
#
# Usage:
#
# >> bash killer.sh celery
#
# or better to alias this script in your .bashrc/.zshrc
# so you can use it like:
#
# >> killer npm
# >> killer celery
# >> killer fuckingJava

victim_name=${1}

if [ "$victim_name" == "" ]
then
    echo "Nope! Gimme a victim name."
    exit
fi

output="$(ps ax | grep ${victim_name} | awk '{print $1,$3}')"
# at this point output looks like this:
# 254214 S
# 254215 S
# 254216 S
# 259206 S+
# 259207 S+

# we change internal field separator to use newline as a separator
_IFS=$IFS
IFS=$'\n'
pid_state_array=($output)
IFS=$_IFS

# pids to be killed
victim_pids=()

for pid_state in "${pid_state_array[@]}"; do
    pid_state=($pid_state)
    # we ignore the current process and its child
    if [ "${pid_state[0]}" != $$ ] && [ "${pid_state[1]}" != "S+" ]
    then
        victim_pids+=("${pid_state[0]}")
    fi
done

if [ "${#victim_pids[@]}" == 0 ]
then
    echo "Nothing found for '${victim_name}'."
    exit
fi

echo "Got them: ${victim_pids[@]}";
echo "$(kill -9 "${victim_pids[@]}" >/dev/null 2>&1)"
echo ".. and smashed!"