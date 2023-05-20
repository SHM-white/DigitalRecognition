### 部署流程
1. 安装gh，在ubuntu上，使用以下指令安装（这是一行指令，请一次性复制并执行）：
    ```shell
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && sudo apt update \
    && sudo apt install gh -y
    ```
2. 执行`gh auth login`登陆github
3. 在model目录下执行`update_model.sh`更新model
4. 如果你需要使用tensorrt后端，请使用train分支下的转换工具自行转换
5. 开始运行