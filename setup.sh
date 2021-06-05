# install required packages

sudo apt-get install xvfb

pip install gym box2d box2d-py
pip install --upgrade h5py==2.10.0
pip install pyvirtualdisplay
pip install Pillow
pip install tqdm

# clone repo from github

rm -rf ./OpenAI-GYM-CarRacing-DQN
rm -rf ./DQN
git clone https://github.com/Mzhhh/OpenAI-GYM-CarRacing-DQN
mv ./OpenAI-GYM-CarRacing-DQN ./DQN
cp ./DQN/save/trial_500.h5 ./tf_best.h5
touch ./DQN/__init__.py

rm -rf ./TD3
git clone https://github.com/Mzhhh/TD3.git
touch ./TD3/__init__.py


