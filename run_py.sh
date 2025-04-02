#pip install humenv/.
rm -r videos*

#mkdir -p ~/anaconda3/lib.backup && cp ~/anaconda3/lib/libstdc++.so.6* ~/anaconda3/lib.backup/ 2>/dev/null || echo "No files to backup"
#ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30 ~/anaconda3/lib/libstdc++.so.6

xvfb-run -a python forward_reward_demo.py --model-name="facebook/metamotivo-M-1" --camera=front --move-speed=1.0 --num-steps=1000 --video-dir videos/정상_1/정면
xvfb-run -a python forward_reward_demo.py --model-name="facebook/metamotivo-M-1" --camera=side --move-speed=1.0 --num-steps=1000 --video-dir videos/정상_1/측면
xvfb-run -a python forward_reward_demo.py --model-name="facebook/metamotivo-M-1" --camera=front --move-speed=2.0 --num-steps=1000 --video-dir videos/정상_2/정면
xvfb-run -a python forward_reward_demo.py --model-name="facebook/metamotivo-M-1" --camera=side --move-speed=2.0 --num-steps=1000 --video-dir videos/정상_2/측면
xvfb-run -a python forward_reward_demo.py --model-name="facebook/metamotivo-M-1" --camera=front --move-speed=3.0 --num-steps=1000 --video-dir videos/정상_3/정면
xvfb-run -a python forward_reward_demo.py --model-name="facebook/metamotivo-M-1" --camera=side --move-speed=3.0 --num-steps=1000 --video-dir videos/정상_3/측면
xvfb-run -a python forward_reward_demo.py --model-name="facebook/metamotivo-M-1" --unbalanced --camera=front --move-speed=1.0 --num-steps=1000 --video-dir videos/언밸런스_학습1/정면
xvfb-run -a python forward_reward_demo.py --model-name="facebook/metamotivo-M-1" --unbalanced --camera=side --move-speed=1.0 --num-steps=1000 --video-dir videos/언밸런스_학습1/측면
xvfb-run -a python forward_reward_demo.py --model-name="facebook/metamotivo-M-1" --unbalanced --camera=front --move-speed=2.0 --num-steps=1000 --video-dir videos/언밸런스_학습2/정면
xvfb-run -a python forward_reward_demo.py --model-name="facebook/metamotivo-M-1" --unbalanced --camera=side --move-speed=2.0 --num-steps=1000 --video-dir videos/언밸런스_학습2/측면
xvfb-run -a python forward_reward_demo.py --model-name="facebook/metamotivo-M-1" --unbalanced --camera=front --move-speed=3.0 --num-steps=1000 --video-dir videos/언밸런스_학습3/정면
xvfb-run -a python forward_reward_demo.py --model-name="facebook/metamotivo-M-1" --unbalanced --camera=side --move-speed=3.0 --num-steps=1000 --video-dir videos/언밸런스_학습3/측면
