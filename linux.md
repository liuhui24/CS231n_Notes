Linux & Vim 学习笔记
 
markdown
  
# Linux & Vim 学习笔记
> 适用环境：Ubuntu / WSL2，CV科研实操专用
 
 
一、基础目录路径命令
 
命令 英文全称 功能说明 示例 
 pwd  Print Working Directory 打印当前工作目录  pwd  
 cd  Change Directory 切换目录  cd ~ 、 cd ..  
 ls  List 罗列目录文件  ls -l 、 ls -a  
 mkdir  Make Directory 创建文件夹  mkdir linux_learn  
 rmdir  Remove Directory 删除空文件夹  rmdir test  
 
二、文件增删改移
 
命令 英文全称 功能说明 示例 
 touch  touch 创建空文件  touch info.txt  
 cp  Copy 复制文件/目录  cp info.txt bak.txt  
 mv  Move 移动、重命名  mv bak.txt ./linux_learn/  
 rm  Remove 删除文件  rm info.txt 、 rm -rf 目录  
 
三、文件查看命令
 
命令 英文全称 功能说明 示例 
 cat  Concatenate 全量查看文件  cat info.txt  
 head  head 查看文件前N行  head -n5 info.txt  
 tail  tail 查看文件末尾N行  tail -n5 info.txt  
 
四、权限管理
 
命令 英文全称 功能说明 示例 
 chmod  Change Mode 修改文件权限  chmod +x run.sh  
 chown  Change Owner 修改文件归属用户  chown user test.txt  
 
五、查找 + 文本三剑客（CV看日志高频）
 
命令 英文全称 功能说明 示例 
 find  find 磁盘检索文件  find ~ -name "*.py"  
 grep  Global Regular Expression Print 正则文本检索  grep "loss" train.log  
 sed  Stream Editor 流式文本替换  sed 's/old/new/g' test.txt  
 awk  Aho-Weinberger-Kernighan 按列截取文本  awk '{print $3}' log.txt  
 
六、系统&软件管理
 
命令 英文全称 功能说明 示例 
 sudo  Super User Do 管理员权限运行  sudo apt install tree  
 apt  Advanced Packaging Tool Debian软件管理器  sudo apt update  
 df  Disk Free 磁盘使用率查看  df -h  
 free  free 内存占用查看  free -h  
 top  table of processes 实时进程监控  top  
 
七、压缩 & 远程连接
 
命令 英文全称 功能说明 示例 
 tar  Tape Archive 归档压缩解压  tar -zcvf out.tar.gz file  
 zip  Zip 打包zip  zip code.zip *.py  
 unzip  Un-Zip 解压zip  unzip code.zip  
 ssh  Secure Shell 远程登录服务器  ssh user@ip  
 scp  Secure Copy 远程传输文件  scp test.py user@ip:/home  
 
八、重定向、管道符
 
符号/命令 全称释义 作用 示例 
 echo  echo 控制台输出、写入文本  echo "loss=0.25" >> log.txt  
 >  redirect output 覆盖写入文件  echo 123 > tmp.txt  
 >>  append redirect 追加写入  echo 456 >> tmp.txt  
 |  pipe 管道串联命令  cat log.txt | grep loss  
 
九、用户管理
 
命令 英文全称 功能 
 useradd  User Add 新建系统用户 
 userdel  User Delete 删除系统用户 
 passwd  Password 修改用户密码 
 
十、常用补充指令
 
命令 英文全称 功能 
 stat  Status 查看文件详细属性 
 tree  tree 树形展示目录 
 ps  Process Status 瞬时进程查看 
 kill  Kill 终止进程， kill -9 强制杀死 
 env  Environment 查看全部环境变量 
 export  export 临时配置环境变量 
 diff  Difference 对比两个文件差异 
 
Vim 操作手册
 
模式说明
 
- Normal：默认普通模式， Esc(escape) 从插入切回
- Insert：插入编辑模式
- Command：底行命令模式，输入 : 唤起
 
1. 切入插入模式
 
按键 英文释义 功能 
 i  insert 光标前插入 
 I  Insert 当前行首插入 
 a  append 光标后追加 
 A  Append 当前行尾追加 
 o  open 光标下方新开一行 
 O  Open 光标上方新开一行 
 
2. 光标跳转
 
按键 释义 功能 
 h  left 左移 
 j  down 下移 
 k  up 上移 
 l  right 右移 
 w  word 下个单词开头 
 b  backword 上个单词开头 
 0  zero 跳到行首 
 $  end of line 跳到行尾 
 gg  go global 文件首行 
 G  Go 文件末行 
 nG  n Go 跳转至第n行 
 
3. 删除指令(d=delete)
 
指令 全称 功能 
 x  cut char 删除单个字符 
 dd  delete line 删除整行 
 dw  delete word 删除光标后一个单词 
 d$  delete to end 删除至行末尾 
 
4. 复制粘贴(y=yank / p=put)
 
指令 全称 功能 
 yy  yank line 复制整行 
 yw  yank word 复制单个单词 
 p  put 光标下方粘贴 
 P  Put 光标上方粘贴 
 
5. 撤销与重做
 
按键 全称 功能 
 u  undo 撤销 
 Ctrl+r  redo 重做 
 
6. 底行命令
 
命令 全称 功能 
 :w  write 保存 
 :q  quit 退出 
 :wq  write+quit 保存并退出 
 :q!  quit force 强制不保存退出 
 :set nu  set number 开启行号 
 :set nonu  no number 关闭行号 
 
7. 文本搜索
 
指令 作用 
 /关键词  向后查找， n 下一个， N 上一个 
