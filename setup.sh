alias py=python3
pip install -r requirements.txt & PIDPIP=$!
apt-get update && apt-get install unzip & PIDAPT=$!
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1qVfVh2gGTG742C6J0l9WB-aBuJkcb1va' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1qVfVh2gGTG742C6J0l9WB-aBuJkcb1va" -O data_v2.zip && rm -rf /tmp/cookies.txt & PIDGET=$!
wait $PIDAPT
apt-get install tmux vim ffmpeg libsm6 libxext6  -y &
wait $PIDGET
unzip data_v2.zip
mkdir runs
wait $PIDPIP
wait # for miscellaneous apt packages to install
