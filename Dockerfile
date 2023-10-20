# 指定基础镜像
FROM torch1.12.1_cuda11.3_python3.10.0:1.0
# 指定镜像的维护者信息，一般为邮箱
MAINTAINER yanwj "yanwj@finchina.com"
# 构建镜像
RUN mkdir table
# 拷贝文件
COPY / /table
# 设置当前工作目录
WORKDIR /table
# 指定容器启动时需要运行的命令 ENTRYPOINT命令可以追加命令
# RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
ENTRYPOINT [ "python" ]
# 指定容器启动的时候需要执行的命令 只有最后一个命令会生效
CMD [ "server.py" ]