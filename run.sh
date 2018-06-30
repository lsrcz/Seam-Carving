echo -e "\x1b[34;1mcolumn first\x1b[0m"
#python3 seam_main.py --order=col ./pics/ori.jpg 212 300 0 ./col.jpg
echo -e "\x1b[34;1mrow first\x1b[0m"
#python3 seam_main.py --order=row ./pics/ori.jpg 212 300 0 ./row.jpg
echo -e "\x1b[34;1mincrease\x1b[0m"
#python3 seam_main.py ./pics/fuji.jpg 998 1431 0 ./increase.jpg
echo -e "\x1b[34;1mwithout local entropy\x1b[0m"
#python3 seam_main.py ./pics/Broadway_tower_edit.jpg 248 371 0 ./noentropy.jpg
echo -e "\x1b[34;1mwith local entropy\x1b[0m"
#python3 seam_main.py --ratio=0.1 ./pics/Broadway_tower_edit.jpg 248 371 1 ./entropy.jpg
echo -e "\x1b[34;1mwith forward energy\x1b[0m"
#python3 seam_main.py ./pics/car3.jpg 700 562 2 ./forward.jpg
echo -e "\x1b[34;1mdensenet CAM\x1b[0m"
python3 seam_main.py --net='densenet' ./pics/car3.jpg 700 562 3 ./densenet.jpg
echo -e "\x1b[34;1msqueezenet CAM\x1b[0m"
python3 seam_main.py --net='squeezenet' ./pics/car3.jpg 700 562 3 ./squeezenet.jpg
echo -e "\x1b[34;1mdensenet CAM with forward\x1b[0m"
python3 seam_main.py --net='densenet' ./pics/car3.jpg 1200 562 4 ./densenet_fwd.jpg
echo -e "\x1b[34;1msqueezenet CAM with forward\x1b[0m"
python3 seam_main.py --net='squeezenet' ./pics/car3.jpg 1200 562 4 ./squeezenet_fwd.jpg
echo -e "\x1b[34;1moptimal seam order\x1b[0m"
python3 seam_main.py --order=opt ./pics/ori.jpg 212 300 0 ./opt.jpg