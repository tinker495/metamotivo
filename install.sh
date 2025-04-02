apt install xvfb -y -qq
python -m venv venv
source venv/bin/activate


pip install -r requirements.txt
pip install .
git clone https://github.com/tinker495/humenv.git
pip install humenv/.

wget https://huggingface.co/datasets/Tinker/testing_motivo/resolve/main/datas.zip
unzip datas.zip
rm datas.zip

wget https://huggingface.co/datasets/Tinker/testing_motivo/resolve/main/motivo-S-unbalanced.zip
unzip motivo-S-unbalanced.zip
rm motivo-S-unbalanced.zip