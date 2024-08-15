# Osu

todo list:

- [ ] add a readme file

use 4 frame to an input to the model and get the output.

- [ ] how to get frame?

-------------1234
----------1-2-3-4
-------1--2--3--4
----1---2---3---4
-1----2----3----4

below 223 = 0

- [ ] 224 to 255 = 0 to 255

issues:
經驗回放區中太多0獎勵 應全部取出或僅留下極少部分
好好檢查模型處理數據的方式 一個錯就容易全錯

reward 往前推:
點下原圈之後 才會得到獎勵 但圓圈已經消失
所以須將獎勵往前移動 才能引導ai正確移動