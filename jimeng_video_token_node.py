"""
即梦AI图生视频节点（使用Token直接调用国内端点）
支持文生视频和图生视频，直接使用Token，无需号池
"""

import os
import json
import logging
import torch
import numpy as np
import time
import requests
import io
import uuid
import hashlib
import hmac
import datetime
import urllib.parse
import tempfile
import shutil
import cv2
from PIL import Image
from typing import Dict, Any, Tuple, Optional, List

# 导入核心模块
from .core.token_manager import TokenManager
from .core.api_client import ApiClient

# 兼容性导入
try:
    import folder_paths
except ImportError:
    # 如果无法导入，使用自定义路径
    class FolderPaths:
        @staticmethod
        def get_output_directory():
            return os.path.join(os.path.dirname(__file__), "output")
    folder_paths = FolderPaths()

try:
    from comfy.comfy_types import IO
except ImportError:
    # 如果无法导入，使用字符串类型
    class IOCompat:
        VIDEO = "VIDEO"
    IO = IOCompat()

logger = logging.getLogger(__name__)


class JimengVideoAdapter:
    """
    视频适配器，封装视频路径，使其能被ComfyUI的保存视频节点识别
    """
    def __init__(self, video_path: str):
        self.video_path = video_path
    
    def get_dimensions(self):
        """获取视频的宽度和高度"""
        try:
            if not self.video_path or not os.path.exists(self.video_path):
                return 1280, 720  # 默认值
            cap = cv2.VideoCapture(self.video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return width, height
        except Exception as e:
            logger.error(f"[JimengVideoAdapter] 获取视频尺寸失败: {e}")
            return 1280, 720
    
    def save_to(self, output_path, format="auto", codec="auto", metadata=None):
        """保存视频到指定路径"""
        try:
            if self.video_path and os.path.exists(self.video_path):
                shutil.copyfile(self.video_path, output_path)
                return True
            else:
                logger.error(f"[JimengVideoAdapter] 源视频文件路径无效: {self.video_path}")
                return False
        except Exception as e:
            logger.error(f"[JimengVideoAdapter] 保存视频时出错: {e}")
            return False


def _load_config_for_class() -> Dict[str, Any]:
    """
    辅助函数：用于在节点类实例化前加载配置
    """
    try:
        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(plugin_dir, "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"[JimengVideoNode] 无法加载配置文件: {e}。将使用默认值。")
        return {"params": {}, "timeout": {}}


class JimengVideoTokenNode:
    """
    即梦AI图生视频节点
    通过Token直接调用国内即梦API，支持文生视频和图生视频
    """
    def __init__(self):
        self.plugin_dir = os.path.dirname(os.path.abspath(__file__))
        self.config = self._load_config()
        self.base_url = "https://jimeng.jianying.com"
        self.aid = "513695"
        self.app_version = "8.4.0"  # 更新为真实版本
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = "jimeng_video"
        
        # 模型名称映射
        self.model_map = {
            "jimeng-video-3.0-pro": "dreamina_ic_generate_video_model_vgfm_3.0_pro",
            "jimeng-video-3.0-fast": "dreamina_ic_generate_video_model_vgfm_3.0_fast",
            "jimeng-video-3.0": "dreamina_ic_generate_video_model_vgfm_3.0_fast",
            "jimeng-video-2.0": "dreamina_ic_generate_video_model_vgfm_2.0"
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """加载插件的 config.json 配置文件"""
        try:
            config_path = os.path.join(self.plugin_dir, "config.json")
            if not os.path.exists(config_path):
                template_path = os.path.join(self.plugin_dir, "config.json.template")
                if os.path.exists(template_path):
                    shutil.copy(template_path, config_path)
                    logger.info("[JimengVideoNode] 从模板创建了 config.json")
                else:
                    logger.warning("[JimengVideoNode] 配置文件和模板文件都不存在！")
                    return {"params": {}, "timeout": {}}
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info("[JimengVideoNode] 配置文件加载成功")
            return config
        except Exception as e:
            logger.error(f"[JimengVideoNode] 配置文件加载失败: {e}")
            return {"params": {}, "timeout": {}}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "一只可爱的小猫在街上行走",
                    "tooltip": "视频描述文本"
                }),
                "token": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "请输入即梦sessionid"
                }),
                "model": ([
                    "jimeng-video-3.0-fast",
                    "jimeng-video-3.0-pro",
                    "jimeng-video-2.0"
                ], {
                    "default": "jimeng-video-3.0-fast",
                    "tooltip": "生成模型"
                }),
                "aspect_ratio": ([
                    "21:9",
                    "16:9",
                    "4:3",
                    "1:1",
                    "3:4",
                    "9:16"
                ], {
                    "default": "9:16",
                    "tooltip": "视频比例"
                }),
                "resolution": ([
                    "720p",
                    "1080p"
                ], {
                    "default": "720p",
                    "tooltip": "视频分辨率"
                }),
                "duration": ([
                    "5s",
                    "10s"
                ], {
                    "default": "5s",
                    "tooltip": "视频时长（10s需要积分≥90）"
                }),
            },
            "optional": {
                "first_frame": ("IMAGE", {
                    "tooltip": "可选：首帧图片，不输入则为文生视频"
                }),
                "end_frame": ("IMAGE", {
                    "tooltip": "可选：尾帧图片"
                }),
            }
        }
    
    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "info")
    FUNCTION = "generate_video"
    OUTPUT_NODE = False
    CATEGORY = "即梦AI"
    
    def calculate_dimensions(self, aspect_ratio, resolution):
        """根据比例和分辨率计算宽高"""
        # 基准高度
        base_height = 720 if resolution == "720p" else 1080
        
        # 比例映射到宽高
        ratio_map = {
            "21:9": (21, 9),
            "16:9": (16, 9),
            "4:3": (4, 3),
            "1:1": (1, 1),
            "3:4": (3, 4),
            "9:16": (9, 16)
        }
        
        w_ratio, h_ratio = ratio_map.get(aspect_ratio, (16, 9))
        
        # 根据比例计算宽度
        if h_ratio >= w_ratio:
            # 竖屏或方形，高度为基准
            height = base_height
            width = int(height * w_ratio / h_ratio)
        else:
            # 横屏，宽度为基准
            width = int(base_height * 16 / 9)
            height = int(width * h_ratio / w_ratio)
        
        # 确保是64的倍数（向上取整）
        width = ((width + 63) // 64) * 64
        height = ((height + 63) // 64) * 64
        
        return width, height
    
    def tensor_to_pil(self, tensor):
        """将ComfyUI的tensor转换为PIL图片"""
        img_array = tensor[0].cpu().numpy()
        img_array = (img_array * 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def _get_upload_token(self, token_manager):
        """获取上传token"""
        try:
            url = f"{self.base_url}/mweb/v1/get_upload_token"
            params = {
                "aid": self.aid,
                "device_platform": "web",
                "region": "CN"
            }
            
            # 准备POST请求体
            data = {
                "scene": 2
            }
            
            # 获取token信息
            token_info = token_manager.get_token('/mweb/v1/get_upload_token')
            
            headers = {
                'accept': 'application/json, text/plain, */*',
                'accept-language': 'zh-CN,zh;q=0.9',
                'app-sdk-version': '48.0.0',
                'appid': self.aid,
                'appvr': self.app_version,
                'content-type': 'application/json',
                'cookie': token_info.get("cookie", ""),
                'device-time': token_info.get("device_time", ""),
                'lan': 'zh-Hans',
                'loc': 'cn',
                'origin': self.base_url,
                'pf': '7',
                'priority': 'u=1, i',
                'referer': f'{self.base_url}/ai-tool/video/generate',
                'sign': token_info.get("sign", ""),
                'sign-ver': '1',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
            }
            
            # 发送POST请求
            response = requests.post(url, headers=headers, params=params, json=data)
            response.raise_for_status()
            result = response.json()
            
            if result.get("ret") != "0":
                logger.error(f"[JimengVideoNode] Failed to get upload token: {result}")
                return None
                
            data = result.get("data", {})
            if not data:
                logger.error("[JimengVideoNode] No data in get_upload_token response")
                return None
                
            return data
        except Exception as e:
            logger.error(f"[JimengVideoNode] Error getting upload token: {e}")
            return None
    
    def _upload_image_for_video(self, image_path, upload_token):
        """上传图片用于视频生成，返回(uri, width, height, format)"""
        try:
            # 获取图片尺寸信息
            with Image.open(image_path) as img:
                img_width, img_height = img.size
                img_format = img.format.lower() if img.format else 'png'
            
            # 获取文件大小
            file_size = os.path.getsize(image_path)
            
            # 第一步：申请图片上传
            t = datetime.datetime.utcnow()
            amz_date = t.strftime('%Y%m%dT%H%M%SZ')
            
            request_parameters = {
                'Action': 'ApplyImageUpload',
                'FileSize': str(file_size),
                'ServiceId': upload_token.get('space_name', 'tb4s082cfz'),
                'Version': '2018-08-01'
            }
            
            canonical_querystring = '&'.join([f'{k}={urllib.parse.quote(str(v))}' for k, v in sorted(request_parameters.items())])
            
            canonical_uri = '/'
            canonical_headers = (
                f'host:imagex.bytedanceapi.com\n'
                f'x-amz-date:{amz_date}\n'
                f'x-amz-security-token:{upload_token.get("session_token", "")}\n'
            )
            signed_headers = 'host;x-amz-date;x-amz-security-token'
            
            payload_hash = hashlib.sha256(b'').hexdigest()
            
            canonical_request = '\n'.join([
                'GET',
                canonical_uri,
                canonical_querystring,
                canonical_headers,
                signed_headers,
                payload_hash
            ])
            
            authorization = self._get_aws_authorization(
                upload_token.get('access_key_id', ''),
                upload_token.get('secret_access_key', ''),
                'cn-north-1',
                'imagex',
                amz_date,
                upload_token.get('session_token', ''),
                signed_headers,
                canonical_request
            )
            
            headers = {
                'Authorization': authorization,
                'X-Amz-Date': amz_date,
                'X-Amz-Security-Token': upload_token.get('session_token', ''),
                'Host': 'imagex.bytedanceapi.com'
            }
            
            url = f'https://imagex.bytedanceapi.com/?{canonical_querystring}'
            response = requests.get(url, headers=headers)
            
            if response.status_code != 200:
                logger.error(f"[JimengVideoNode] Failed to get upload authorization: {response.text}")
                return None
                
            upload_info = response.json()
            if not upload_info or "Result" not in upload_info:
                logger.error(f"[JimengVideoNode] No Result in ApplyImageUpload response")
                return None
            
            # 第二步：上传图片文件
            store_info = upload_info['Result']['UploadAddress']['StoreInfos'][0]
            upload_host = upload_info['Result']['UploadAddress']['UploadHosts'][0]
            
            url = f"https://{upload_host}/upload/v1/{store_info['StoreUri']}"
            
            # 计算文件的CRC32
            with open(image_path, 'rb') as f:
                content = f.read()
                import binascii
                crc32 = format(binascii.crc32(content) & 0xFFFFFFFF, '08x')
            
            headers = {
                'accept': '*/*',
                'authorization': store_info['Auth'],
                'content-type': 'application/octet-stream',
                'content-disposition': 'attachment; filename="undefined"',
                'content-crc32': crc32,
                'origin': self.base_url,
                'referer': f'{self.base_url}/'
            }
            
            response = requests.post(url, headers=headers, data=content)
            if response.status_code != 200:
                logger.error(f"[JimengVideoNode] Failed to upload image: {response.text}")
                return None
                
            upload_result = response.json()
            if upload_result.get("code") != 2000:
                logger.error(f"[JimengVideoNode] Upload image error: {upload_result}")
                return None
            
            # 第三步：提交上传
            session_key = upload_info['Result']['UploadAddress']['SessionKey']
            store_uri = store_info.get("StoreUri", "")
            
            amz_date = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            
            params = {
                "Action": "CommitImageUpload",
                "Version": "2018-08-01",
                "ServiceId": upload_token.get('space_name', 'tb4s082cfz')
            }
            
            commit_data = {
                "SessionKey": session_key,
                "UploadHosts": upload_info['Result']['UploadAddress']['UploadHosts'],
                "StoreKeys": [store_uri]
            }
            
            payload = json.dumps(commit_data)
            content_sha256 = hashlib.sha256(payload.encode('utf-8')).hexdigest()
            
            canonical_uri = "/"
            canonical_querystring = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
            signed_headers = "x-amz-content-sha256;x-amz-date;x-amz-security-token"
            canonical_headers = f"x-amz-content-sha256:{content_sha256}\nx-amz-date:{amz_date}\nx-amz-security-token:{upload_token.get('session_token', '')}\n"
            
            canonical_request = f"POST\n{canonical_uri}\n{canonical_querystring}\n{canonical_headers}\n{signed_headers}\n{content_sha256}"
            
            authorization = self._get_aws_authorization(
                upload_token.get('access_key_id', ''),
                upload_token.get('secret_access_key', ''),
                'cn-north-1',
                'imagex',
                amz_date,
                upload_token.get('session_token', ''),
                signed_headers,
                canonical_request
            )
            
            headers = {
                'accept': '*/*',
                'content-type': 'application/json',
                'authorization': authorization,
                'x-amz-content-sha256': content_sha256,
                'x-amz-date': amz_date,
                'x-amz-security-token': upload_token.get('session_token', ''),
                'origin': self.base_url,
                'referer': f'{self.base_url}/'
            }
            
            commit_url = f"https://{upload_token.get('upload_domain', 'imagex.bytedanceapi.com')}"
            response = requests.post(f"{commit_url}?{canonical_querystring}", headers=headers, data=payload)
            
            if response.status_code != 200:
                logger.error(f"[JimengVideoNode] Failed to commit upload: {response.text}")
                return None
                
            commit_result = response.json()
            if not commit_result or "Result" not in commit_result:
                logger.error(f"[JimengVideoNode] No Result in CommitImageUpload response")
                return None
            
            # 返回图片URI和元数据（按照真实API格式）
            return {
                "type": "image",
                "id": str(uuid.uuid4()),
                "source_from": "upload",
                "platform_type": 1,
                "name": "",
                "image_uri": store_uri,
                "width": img_width,
                "height": img_height,
                "format": img_format or "",
                "uri": store_uri
            }
            
        except Exception as e:
            logger.error(f"[JimengVideoNode] Error uploading image: {e}")
            return None
    
    def _get_aws_authorization(self, access_key, secret_key, region, service, amz_date, security_token, signed_headers, canonical_request):
        """获取AWS V4签名授权头"""
        try:
            datestamp = amz_date[:8]
            
            canonical_request_hash = hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()
            
            credential_scope = f"{datestamp}/{region}/{service}/aws4_request"
            string_to_sign = f"AWS4-HMAC-SHA256\n{amz_date}\n{credential_scope}\n{canonical_request_hash}"
            
            k_date = hmac.new(f"AWS4{secret_key}".encode('utf-8'), datestamp.encode('utf-8'), hashlib.sha256).digest()
            k_region = hmac.new(k_date, region.encode('utf-8'), hashlib.sha256).digest()
            k_service = hmac.new(k_region, service.encode('utf-8'), hashlib.sha256).digest()
            k_signing = hmac.new(k_service, b'aws4_request', hashlib.sha256).digest()
            
            signature = hmac.new(k_signing, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
            
            authorization = (
                f"AWS4-HMAC-SHA256 Credential={access_key}/{credential_scope}, "
                f"SignedHeaders={signed_headers}, Signature={signature}"
            )
            
            return authorization
        except Exception as e:
            logger.error(f"[JimengVideoNode] Error generating authorization: {str(e)}")
            return ""
    
    def _generate_video_api(self, token_manager, prompt, model, width, height, resolution, duration, aspect_ratio, first_frame_info=None, end_frame_info=None):
        """调用视频生成API（按照真实API结构）"""
        try:
            url = f"{self.base_url}/mweb/v1/aigc_draft/generate"
            
            # 生成唯一ID
            submit_id = str(uuid.uuid4())
            draft_id = str(uuid.uuid4())
            component_id = str(uuid.uuid4())
            video_gen_input_id = str(uuid.uuid4())
            
            # 获取真实模型名称
            model_req_key = self.model_map.get(model, "dreamina_ic_generate_video_model_vgfm_3.0_fast")
            
            # 转换duration为毫秒
            duration_ms = int(duration.replace('s', '')) * 1000
            
            # 生成随机seed
            import random
            seed = random.randint(10000000, 99999999)
            
            # 构建video_gen_input
            video_gen_input = {
                "type": "",
                "id": video_gen_input_id,
                "min_version": "3.0.5",
                "prompt": prompt,
                "video_mode": 2,
                "fps": 24,
                "duration_ms": duration_ms,
                "resolution": resolution,
                "idip_meta_list": []
            }
            
            # 如果有首帧图片，添加first_frame_image
            if first_frame_info:
                video_gen_input["first_frame_image"] = first_frame_info
            
            # 如果有尾帧图片，添加last_frame_image
            if end_frame_info:
                video_gen_input["last_frame_image"] = end_frame_info
            
            # 构建component（按照真实API结构）
            component = {
                "type": "video_base_component",
                "id": component_id,
                "min_version": "1.0.0",
                "aigc_mode": "workbench",
                "metadata": {
                    "type": "",
                    "id": str(uuid.uuid4()),
                    "created_platform": 3,
                    "created_platform_version": "",
                    "created_time_in_ms": str(int(time.time() * 1000)),
                    "created_did": ""
                },
                "generate_type": "gen_video",  # 关键！不是i2v/t2v
                "abilities": {
                    "type": "",
                    "id": str(uuid.uuid4()),
                    "gen_video": {  # 关键！不是generate
                        "type": "",
                        "id": str(uuid.uuid4()),
                        "text_to_video_params": {
                            "type": "",
                            "id": str(uuid.uuid4()),
                            "video_gen_inputs": [video_gen_input],
                            "video_aspect_ratio": aspect_ratio,
                            "seed": seed,
                            "model_req_key": model_req_key,
                            "priority": 0
                        },
                        "video_task_extra": json.dumps({
                            "promptSource": "custom",
                            "isDefaultSeed": 1,
                            "originSubmitId": submit_id,
                            "isRegenerate": False,
                            "enterFrom": "click",
                            "functionMode": "first_last_frames" if first_frame_info or end_frame_info else "text_to_video"
                        })
                    },
                    "process_type": 1
                }
            }
            
            # 构建draft_content
            draft_content = {
                "type": "draft",
                "id": draft_id,
                "min_version": "3.0.5",
                "min_features": [],
                "is_from_tsn": True,
                "version": "3.3.3",
                "main_component_id": component_id,
                "component_list": [component]
            }
            
            # 准备请求数据（按照真实API格式）
            data = {
                "extend": {
                    "root_model": model_req_key,
                    "m_video_commerce_info": {
                        "benefit_type": "basic_video_operation_vgfm_v_three",
                        "resource_id": "generate_video",
                        "resource_id_type": "str",
                        "resource_sub_type": "aigc"
                    },
                    "m_video_commerce_info_list": [{
                        "benefit_type": "basic_video_operation_vgfm_v_three",
                        "resource_id": "generate_video",
                        "resource_id_type": "str",
                        "resource_sub_type": "aigc"
                    }]
                },
                "submit_id": submit_id,
                "metrics_extra": json.dumps({
                    "promptSource": "custom",
                    "isDefaultSeed": 1,
                    "originSubmitId": submit_id,
                    "isRegenerate": False,
                    "enterFrom": "click",
                    "functionMode": "first_last_frames" if first_frame_info or end_frame_info else "text_to_video"
                }),
                "draft_content": json.dumps(draft_content),
                "http_common_info": {"aid": self.aid}
            }
            
            # 获取token信息
            token_info = token_manager.get_token('/mweb/v1/aigc_draft/generate')
            
            params = {
                "aid": self.aid,
                "device_platform": "web",
                "region": "cn",
                "webId": token_manager.get_web_id(),
                "da_version": "3.3.3",
                "web_component_open_flag": "1",
                "web_version": "7.5.0",
                "aigc_features": "app_lip_sync"
            }
            
            if token_info.get("msToken"):
                params["msToken"] = token_info["msToken"]
            if token_info.get("a_bogus"):
                params["a_bogus"] = token_info["a_bogus"]
            
            headers = {
                'accept': 'application/json, text/plain, */*',
                'accept-language': 'zh-CN,zh;q=0.9',
                'app-sdk-version': '48.0.0',
                'appid': self.aid,
                'appvr': self.app_version,
                'content-type': 'application/json',
                'cookie': token_info.get("cookie", ""),
                'device-time': token_info.get("device_time", ""),
                'lan': 'zh-Hans',
                'loc': 'cn',
                'origin': self.base_url,
                'pf': '7',
                'priority': 'u=1, i',
                'referer': f'{self.base_url}/ai-tool/video/generate',
                'sign': token_info.get("sign", ""),
                'sign-ver': '1',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
            }
            
            # 发送请求
            logger.info(f"[JimengVideoNode] 正在调用API: {url}")
            logger.info(f"[JimengVideoNode] draft_content: {json.dumps(draft_content, ensure_ascii=False)[:500]}...")
            
            response = requests.post(url, headers=headers, params=params, json=data, timeout=900)
            
            logger.info(f"[JimengVideoNode] API响应状态码: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"[JimengVideoNode] API error {response.status_code}: {response.text}")
                return None
                
            result = response.json()
            logger.info(f"[JimengVideoNode] API响应: {json.dumps(result, ensure_ascii=False)[:500]}...")
            
            if str(result.get('ret')) != '0':
                logger.error(f"[JimengVideoNode] Failed to generate video: {result}")
                return None
                
            # 获取history_id用于轮询
            history_id = result.get('data', {}).get('aigc_data', {}).get('history_record_id')
            
            if not history_id:
                logger.error("[JimengVideoNode] No history_id in response")
                return None
            
            logger.info(f"[JimengVideoNode] 视频生成任务已提交，history_id: {history_id}")
            
            return {
                "history_id": history_id,
                "submit_id": submit_id
            }
            
        except Exception as e:
            logger.error(f"[JimengVideoNode] Error generating video: {e}")
            logger.exception(f"[JimengVideoNode] 详细错误堆栈:")
            return None
    
    def _poll_video_status(self, token_manager, history_id):
        """轮询视频生成状态"""
        try:
            url = f"{self.base_url}/mweb/v1/get_history_by_ids"
            
            token_info = token_manager.get_token('/mweb/v1/get_history_by_ids')
            
            params = {
                "aid": self.aid,
                "device_platform": "web",
                "region": "cn",
                "webId": token_manager.get_web_id(),
                "da_version": "3.3.2"
            }
            
            if token_info.get("msToken"):
                params["msToken"] = token_info["msToken"]
            if token_info.get("a_bogus"):
                params["a_bogus"] = token_info["a_bogus"]
            
            data = {
                "history_ids": [history_id]
            }
            
            headers = {
                'accept': 'application/json, text/plain, */*',
                'accept-language': 'zh-CN,zh;q=0.9',
                'app-sdk-version': '48.0.0',
                'appid': self.aid,
                'appvr': self.app_version,
                'content-type': 'application/json',
                'cookie': token_info.get("cookie", ""),
                'device-time': token_info.get("device_time", ""),
                'lan': 'zh-Hans',
                'loc': 'cn',
                'origin': self.base_url,
                'pf': '7',
                'priority': 'u=1, i',
                'referer': f'{self.base_url}/ai-tool/video/generate',
                'sign': token_info.get("sign", ""),
                'sign-ver': '1',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
            }
            
            response = requests.post(url, headers=headers, params=params, json=data)
            
            if response.status_code != 200:
                logger.error(f"[JimengVideoNode] Poll error {response.status_code}: {response.text}")
                return None
            
            result = response.json()
            
            if result.get("ret") != "0":
                logger.error(f"[JimengVideoNode] Poll failed: {result}")
                return None
            
            history_data = result.get("data", {}).get(history_id, {})
            status = history_data.get("status")
            
            # 添加详细调试日志
            logger.info(f"[JimengVideoNode] 轮询状态: status={status}, history_id={history_id}")
            logger.info(f"[JimengVideoNode] 完整响应数据: {json.dumps(history_data, ensure_ascii=False)[:1000]}...")
            
            # 状态50表示完成
            if status == 50:
                # 尝试多种可能的路径提取视频URL
                logger.info(f"[JimengVideoNode] 任务已完成，开始提取视频URL")
                
                # 方法1: 从item_list中提取（真实API结构）
                item_list = history_data.get("item_list", [])
                logger.info(f"[JimengVideoNode] item_list数量: {len(item_list)}")
                
                if item_list and len(item_list) > 0:
                    item = item_list[0]
                    logger.info(f"[JimengVideoNode] item keys: {list(item.keys())}")
                    
                    # 提取视频数据
                    video_data = item.get("video", {})
                    if video_data:
                        logger.info(f"[JimengVideoNode] video_data keys: {list(video_data.keys())}")
                        
                        # 尝试从transcoded_video.origin获取
                        transcoded = video_data.get("transcoded_video", {})
                        if transcoded:
                            origin = transcoded.get("origin", {})
                            video_url = origin.get("video_url")
                            if video_url:
                                logger.info(f"[JimengVideoNode] ✅ 找到视频URL（方法1-transcoded）: {video_url[:100]}...")
                                return video_url
                        
                        # 尝试直接从video_data获取video_url
                        video_url = video_data.get("video_url")
                        if video_url:
                            logger.info(f"[JimengVideoNode] ✅ 找到视频URL（方法1-direct）: {video_url[:100]}...")
                            return video_url
                
                # 方法2: 从resources中提取（备用）
                resources = history_data.get("resources", [])
                logger.info(f"[JimengVideoNode] resources数量: {len(resources)}")
                
                for idx, resource in enumerate(resources):
                    logger.info(f"[JimengVideoNode] resource[{idx}]: type={resource.get('type')}, keys={list(resource.keys())}")
                    if resource.get("type") == "video":
                        video_info = resource.get("video_info", {})
                        video_url = video_info.get("video_url")
                        if video_url:
                            logger.info(f"[JimengVideoNode] ✅ 找到视频URL（方法2）: {video_url[:100]}...")
                            return video_url
                
                # 方法3: 从draft_content中提取（备用）
                draft_content = history_data.get("draft_content")
                if draft_content:
                    try:
                        if isinstance(draft_content, str):
                            draft_obj = json.loads(draft_content)
                        else:
                            draft_obj = draft_content
                        
                        # 遍历component_list查找视频URL
                        component_list = draft_obj.get("component_list", [])
                        for component in component_list:
                            video_url = component.get("video_url")
                            if video_url:
                                logger.info(f"[JimengVideoNode] ✅ 找到视频URL（方法3）: {video_url[:100]}...")
                                return video_url
                    except Exception as e:
                        logger.error(f"[JimengVideoNode] 解析draft_content失败: {e}")
                
                # 方法4: 直接从history_data中提取（备用）
                video_url = history_data.get("video_url")
                if video_url:
                    logger.info(f"[JimengVideoNode] ✅ 找到视频URL（方法4）: {video_url[:100]}...")
                    return video_url
                
                logger.error("[JimengVideoNode] ❌ 所有方法都未找到视频URL")
                logger.error(f"[JimengVideoNode] 完整history_data keys: {list(history_data.keys())}")
                logger.error(f"[JimengVideoNode] 完整history_data: {json.dumps(history_data, ensure_ascii=False)}")
                # 当任务完成但找不到URL时，抛出异常而不是返回None
                raise Exception(f"任务已完成(status=50)但无法提取视频URL，请检查响应数据结构")
            
            # 其他状态表示正在处理（返回None继续轮询）
            logger.debug(f"[JimengVideoNode] 当前状态: {status}, 继续等待...")
            return None
            
        except Exception as e:
            # 如果是我们主动抛出的异常（任务完成但找不到URL），向上传递
            if "任务已完成(status=50)但无法提取视频URL" in str(e):
                logger.error(f"[JimengVideoNode] ❌ 致命错误: {e}")
                raise  # 向上传递异常，停止轮询
            # 其他网络错误等，返回None继续轮询
            logger.error(f"[JimengVideoNode] Error polling video status: {e}")
            return None
    
    def _download_video(self, video_url, prompt):
        """下载视频到本地"""
        try:
            from datetime import datetime
            
            logger.info(f"[JimengVideoNode] 📥 开始下载视频...")
            logger.info(f"[JimengVideoNode] 📥 视频URL: {video_url}")
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_prompt = safe_prompt.replace(' ', '_')
            filename = f"jimeng_video_{timestamp}_{safe_prompt}.mp4"
            
            # 创建输出目录
            output_dir = os.path.join(self.plugin_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(output_dir, filename)
            
            logger.info(f"[JimengVideoNode] 📂 保存路径: {output_path}")
            
            # 下载视频
            logger.info(f"[JimengVideoNode] 📡 正在下载...")
            response = requests.get(video_url, stream=True, timeout=120)
            response.raise_for_status()
            
            # 保存文件
            total_size = 0
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        total_size += len(chunk)
                        # 每5MB输出一次进度
                        if total_size % (5 * 1024 * 1024) < 8192:
                            logger.info(f"[JimengVideoNode] 📊 已下载: {total_size / (1024 * 1024):.2f} MB")
            
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            logger.info(f"[JimengVideoNode] ✅ 视频下载成功！大小: {file_size:.2f} MB")
            logger.info(f"[JimengVideoNode] 💾 文件名: {filename}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"[JimengVideoNode] ❌ 下载视频失败: {e}")
            logger.exception(f"[JimengVideoNode] 详细错误:")
            return None
    
    def generate_video(self, prompt: str, token: str, model: str, aspect_ratio: str, resolution: str, duration: str,
                      first_frame: torch.Tensor = None, end_frame: torch.Tensor = None) -> Tuple[str, str, str]:
        """
        主执行函数：生成视频
        
        Args:
            prompt: 文本提示词
            token: 即梦sessionid
            model: 使用的模型版本
            aspect_ratio: 视频比例
            resolution: 分辨率
            duration: 视频时长
            first_frame: 可选的首帧图像
            end_frame: 可选的尾帧图像
            
        Returns:
            Tuple[str, str, str]: (视频路径, 视频URL, 生成信息)
        """
        try:
            logger.info("=" * 60)
            logger.info("[JimengVideoNode] 即梦AI - 视频生成节点")
            logger.info("=" * 60)
            
            # --- 1. 通用检查 ---
            if not prompt or not prompt.strip():
                return (JimengVideoAdapter(""), "", "错误: 提示词不能为空")
            
            if not token or not token.strip():
                return (JimengVideoAdapter(""), "", "错误: Token不能为空，请输入即梦sessionid")
            
            # --- 2. 使用token创建临时配置和管理器 ---
            temp_config = {
                "accounts": [{"sessionid": token.strip(), "description": "当前Token"}],
                "params": self.config.get("params", {}),
                "timeout": self.config.get("timeout", {})
            }
            
            temp_token_manager = TokenManager(temp_config)
            
            # 检查积分
            required_credit = 90 if duration == "10s" else 45
            current_credit = temp_token_manager.get_credit()
            
            # 显示积分信息
            logger.info(f"[JimengVideoNode] Token积分: {current_credit.get('total_credit', '未知') if current_credit else '未知'}")
            logger.info(f"[JimengVideoNode] 所需积分: {required_credit}")
            
            # 积分检查（如果想强制测试，可以注释掉下面的if语句）
            if not current_credit or current_credit.get('total_credit', 0) < required_credit:
                logger.warning(f"[JimengVideoNode] ⚠️ 警告: 积分可能不足！当前{current_credit.get('total_credit', 0) if current_credit else 0}分，需要{required_credit}分")
                logger.warning(f"[JimengVideoNode] ⚠️ 尝试继续生成，但API可能会拒绝...")
                # return (JimengVideoAdapter(""), "", f"错误: Token积分不足{required_credit}点，无法生成{duration}视频。当前积分: {current_credit.get('total_credit', 0) if current_credit else '未知'}")
                # 注释掉return，允许继续尝试（API会自己检查积分）
            
            # --- 3. 计算视频尺寸 ---
            width, height = self.calculate_dimensions(aspect_ratio, resolution)
            logger.info(f"[JimengVideoNode] 比例: {aspect_ratio} → 尺寸: {width}x{height}")
            
            # --- 4. 判断模式并处理图片上传 ---
            mode = "文生视频"
            first_frame_info = None
            end_frame_info = None
            
            if first_frame is not None or end_frame is not None:
                mode = "图生视频"
                logger.info(f"[JimengVideoNode] 检测到参考图，进入图生视频模式")
                
                # 获取上传token
                upload_token = self._get_upload_token(temp_token_manager)
                if not upload_token:
                    return (JimengVideoAdapter(""), "", "错误: 获取上传token失败")
                
                # 保存并上传首帧
                if first_frame is not None:
                    pil_img = self.tensor_to_pil(first_frame)
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    pil_img.save(temp_file.name, format='PNG')
                    temp_file.close()
                    
                    first_frame_info = self._upload_image_for_video(temp_file.name, upload_token)
                    os.remove(temp_file.name)
                    
                    if not first_frame_info:
                        return (JimengVideoAdapter(""), "", "错误: 首帧图片上传失败")
                    
                    logger.info(f"[JimengVideoNode] 首帧上传成功")
                
                # 保存并上传尾帧
                if end_frame is not None:
                    pil_img = self.tensor_to_pil(end_frame)
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    pil_img.save(temp_file.name, format='PNG')
                    temp_file.close()
                    
                    end_frame_info = self._upload_image_for_video(temp_file.name, upload_token)
                    os.remove(temp_file.name)
                    
                    if not end_frame_info:
                        return (JimengVideoAdapter(""), "", "错误: 尾帧图片上传失败")
                    
                    logger.info(f"[JimengVideoNode] 尾帧上传成功")
            
            logger.info(f"[JimengVideoNode] 模式: {mode}")
            logger.info(f"[JimengVideoNode] 提示词: {prompt[:50]}...")
            logger.info(f"[JimengVideoNode] 模型: {model}")
            logger.info(f"[JimengVideoNode] 分辨率: {resolution}")
            logger.info(f"[JimengVideoNode] 时长: {duration}")
            
            # --- 5. 调用视频生成API ---
            logger.info(f"[JimengVideoNode] 生成中，请耐心等待（可能需要3-15分钟）...")
            logger.info(f"[JimengVideoNode] 开始调用视频生成API，首帧信息: {first_frame_info is not None}, 尾帧信息: {end_frame_info is not None}")
            
            result = self._generate_video_api(
                temp_token_manager, prompt, model, width, height,
                resolution, duration, aspect_ratio, first_frame_info, end_frame_info
            )
            
            logger.info(f"[JimengVideoNode] API调用返回结果: {result}")
            
            if not result:
                logger.error(f"[JimengVideoNode] 视频生成API调用失败，返回None")
                return (JimengVideoAdapter(""), "", "错误: 视频生成API调用失败")
            
            # 获取history_id并确保不为None
            history_id = result.get("history_id")
            if not history_id:
                return (JimengVideoAdapter(""), "", "错误: API未返回有效的任务ID")
            
            # --- 6. 轮询等待视频生成完成 ---
            timeout_config = self.config.get("timeout", {})
            max_wait_time = timeout_config.get("max_wait_time", 900)  # 默认15分钟
            check_interval = timeout_config.get("check_interval", 10)  # 默认10秒间隔
            
            logger.info(f"[JimengVideoNode] ⏰ 开始轮询视频生成状态...")
            logger.info(f"[JimengVideoNode] ⏰ 最大等待时间: {max_wait_time}秒, 检查间隔: {check_interval}秒")
            
            start_time = time.time()
            video_url = None
            poll_count = 0
            
            while time.time() - start_time < max_wait_time:
                poll_count += 1
                elapsed = int(time.time() - start_time)
                
                logger.info(f"[JimengVideoNode] 🔄 第{poll_count}次轮询 (已等待{elapsed}秒)...")
                
                try:
                    video_url = self._poll_video_status(temp_token_manager, history_id)
                    
                    if video_url:
                        logger.info(f"[JimengVideoNode] ✅ 视频生成成功！")
                        logger.info(f"[JimengVideoNode] 🎬 视频URL: {video_url}")
                        break
                    else:
                        logger.info(f"[JimengVideoNode] ⏳ 视频还在生成中，{check_interval}秒后再次检查...")
                except Exception as poll_error:
                    # 任务完成但找不到URL的致命错误
                    logger.error(f"[JimengVideoNode] ❌ 轮询失败: {poll_error}")
                    return (JimengVideoAdapter(""), "", f"错误: {str(poll_error)}")
                
                time.sleep(check_interval)
            
            if not video_url:
                return (JimengVideoAdapter(""), "", f"错误: 等待视频生成超时，已等待 {max_wait_time}秒")
            
            # --- 7. 下载视频 ---
            video_path = self._download_video(video_url, prompt)
            
            if not video_path:
                return (JimengVideoAdapter(""), video_url, "警告: 视频生成成功但下载失败，请使用URL手动下载")
            
            # --- 8. 生成信息文本 ---
            credit_info = temp_token_manager.get_credit()
            info_text = (
                f"🎬 模式: {mode}\n"
                f"🎭 模型: {model}\n"
                f"📐 尺寸: {width}x{height}\n"
                f"📺 分辨率: {resolution}\n"
                f"⏰ 时长: {duration}\n"
                f"💬 提示词: {prompt}\n"
                f"💰 剩余积分: {credit_info.get('total_credit', '未知') if credit_info else '未知'}"
            )
            
            logger.info(f"[JimengVideoNode] 完成！文件: {os.path.basename(video_path)}")
            logger.info(f"[JimengVideoNode] 路径: {video_path}")
            logger.info("=" * 60)
            
            # 返回视频适配器对象、URL和信息
            video_adapter = JimengVideoAdapter(video_path)
            return (video_adapter, video_url, info_text)
            
        except Exception as e:
            logger.exception(f"[JimengVideoNode] 节点执行时发生意外错误")
            return (JimengVideoAdapter(""), "", f"错误: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")


# ComfyUI节点注册
NODE_CLASS_MAPPINGS = {
    "Jimeng_Video_Token": JimengVideoTokenNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Jimeng_Video_Token": "即梦AI视频生成（Token版）"
}

