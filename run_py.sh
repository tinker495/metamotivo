pip install humenv/.
rm -r videos*

#mkdir -p ~/anaconda3/lib.backup && cp ~/anaconda3/lib/libstdc++.so.6* ~/anaconda3/lib.backup/ 2>/dev/null || echo "No files to backup"
#ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30 ~/anaconda3/lib/libstdc++.so.6
xvfb-run -a python forward_reward_demo.py