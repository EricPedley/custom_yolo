pip install -r requirements.txt
apt update
apt install -y tmux
apt install -y unzip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1qVfVh2gGTG742C6J0l9WB-aBuJkcb1va' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1qVfVh2gGTG742C6J0l9WB-aBuJkcb1va" -O data_v2.zip && rm -rf /tmp/cookies.txt
unzip data_v2.zip
echo "alias py=python3" >> ~/.bashrc
source ~/.bashrc
mkdir runs
tmux new-session -d "tensorboard --logdir runs --bind_all"
tmux