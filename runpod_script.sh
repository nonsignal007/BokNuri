## git setting
git config --global user.name "nonsignal007"
git config --global user.email "nonsignal007@gmail.com"

## git ssh setting
touch ~/.ssh/config
echo "Host github.com
    IdentityFile ~/.ssh/runpod_key" >> ~/.ssh/config

## git clone
git clone git@github.com:nonsignal007/LangEyE.git
cd LangEyE

## dependency
sh lang-env.sh
