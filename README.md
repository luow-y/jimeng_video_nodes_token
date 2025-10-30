# 即梦AI视频生成节点（Token版）

## 简介
这是一个ComfyUI自定义节点包，支持即梦AI的视频生成功能。使用Token（sessionid）直接连接国内即梦端点，无需API服务器和号池配置。

## 功能特性
- ✅ 文生视频：通过文本提示词生成视频
- ✅ 图生视频：基于首帧/尾帧图片和提示词生成视频
- ✅ 支持多种视频比例和分辨率
- ✅ 支持5秒和10秒时长
- ✅ 直接使用Token，无需配置
- ✅ 自动下载视频到本地

## 安装方法

### 方法1：直接复制
1. 将整个 `jimeng_video_nodes_token` 文件夹复制到ComfyUI的 `custom_nodes` 目录
2. 重启ComfyUI

### 方法2：Git克隆
```bash
cd ComfyUI/custom_nodes
git clone [你的仓库地址] jimeng_video_nodes_token
cd jimeng_video_nodes_token
pip install -r requirements.txt
```

## 使用方法

### 获取Token（sessionid）
1. 打开即梦官网：https://jimeng.jianying.com
2. 登录您的账号
3. 按F12打开开发者工具
4. 切换到"应用程序"（Application）标签
5. 在左侧找到"Cookies" > "https://jimeng.jianying.com"
6. 复制 `sessionid` 的值

### 在ComfyUI中使用

#### 文生视频
1. 添加 `即梦AI视频生成（Token版）` 节点
2. 在 `token` 框中粘贴您的sessionid
3. 填写提示词
4. 选择模型、比例、分辨率、时长等参数
5. 运行（需要等待3-15分钟）

#### 图生视频
1. 添加 `即梦AI视频生成（Token版）` 节点
2. 在 `token` 框中粘贴您的sessionid
3. 填写提示词
4. 连接首帧图片到 `first_frame` （可选）
5. 连接尾帧图片到 `end_frame` （可选）
6. 运行（需要等待3-15分钟）

## 参数说明

- **prompt**：提示词，描述您想要生成的视频
- **token**：即梦sessionid（必填）
- **model**：选择模型版本（jimeng-video-3.0/2.0）
- **aspect_ratio**：视频比例（21:9, 16:9, 4:3, 1:1, 3:4, 9:16）
- **resolution**：视频分辨率（720p/1080p）
- **duration**：视频时长（5s/10s）
- **first_frame**：可选的首帧图片
- **end_frame**：可选的尾帧图片

## 输出说明

- **video_path**：下载到本地的视频文件路径
- **video_url**：在线视频URL
- **info**：生成信息文本（包含积分信息）

## 积分消耗

- **文生视频（5秒）**：45积分
- **文生视频（10秒）**：90积分
- **图生视频**：额外消耗更多积分

⚠️ **注意**：视频生成比图像生成消耗更多积分，请确保账号有足够积分。

## 视频保存位置
生成的视频默认保存在：
```
ComfyUI/custom_nodes/jimeng_video_nodes_token/output/
```

文件命名格式：
```
jimeng_video_20250130_223045_提示词前30字.mp4
```

## 常见问题

### Q: Token无效
A: 请检查：
- sessionid是否正确复制
- sessionid是否已过期（重新登录获取）
- 网络连接是否正常

### Q: 积分不足
A: 
- 5秒视频需要45积分
- 10秒视频需要90积分
- 请到即梦官网充值

### Q: 生成超时
A: 
- 视频生成时间通常需要3-15分钟
- 可以在 `config.json` 中增加超时时间
- 超时后可以去即梦官网查看生成结果

### Q: 下载失败
A: 
- 检查网络连接
- 检查output目录权限
- 可以手动使用返回的video_url下载

### Q: 图片上传失败
A: 
- 确保图片格式正确（PNG/JPG）
- 检查图片大小（建议不超过10MB）
- 检查网络连接

## 性能提示
- 视频生成是异步的，会自动轮询状态
- 生成过程中会在控制台输出进度
- 可以同时提交多个视频生成任务

## 技术支持
- 查看ComfyUI控制台日志
- 访问项目GitHub Issues

## 许可证
请查看 LICENSE 文件

## 版本历史
- v1.0.0 - 初始版本，支持文生视频和图生视频，直接使用Token连接国内端点



