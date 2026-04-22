import random
import time
import re
import requests
import json
import csv
import os

from playwright.sync_api import sync_playwright
from static.playwright_sign import sign_with_playwright
from static.xhs_sign import get_search_id
from playwright.sync_api import TimeoutError

total_comment_count = 0
note_comment_count = 0

# Setting to suit RedNote
note_type_dict = {
    "全部": 0,
    "视频": 1,
    "图文": 2
}

sort_type_dict = {
    "综合": "general",
    "最新": "time_descending",
    "最多点赞": "popularity_descending",
    "最多评论": "comment_descending",
    "最多收藏": "collect_descending"
}

base_headers = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
    "cache-control": "no-cache",
    "content-type": "application/json;charset=UTF-8",
    "dnt": "1",
    "origin": "https://www.xiaohongshu.com",
    "pragma": "no-cache",
    "priority": "u=1, i",
    "referer": "https://www.xiaohongshu.com/",
    "sec-ch-ua": '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "cookie": "abRequestId=c75477df-c639-5e3d-9da6-346a6c8fa27d; ets=1774555151257; a1=19d2bbabbeaswiabnms93ixm8pfwmenjxo1qdkj1050000396816; webId=8a5ce55ccdbc306460de55582c8e6d89; gid=yjfJDD0SdyCSyjfJDD0DDlTVd0ME30DI6Mjqyy83C6YUiA28E6dIVf888qjKYyK88D2KqjS4; web_session=040069b6f970bf4e6f05efe9f03b4b5646a532; id_token=VjEAAMeqMg2QMoJyWPWVyO1EocTKShkW/qdqZE/fV1isg/vpVJLXpFwJNENTa0c64rAZc173ZtwTnWxeP+RHBTZUhK/4Yv62JoLboeBoyO02BfMHw4aI74CiD5isv4k7iIANYwl2; xsecappid=xhs-pc-web; webBuild=6.3.0; unread={%22ub%22:%2269ac2aba000000001a02a6fa%22%2C%22ue%22:%2269be31e80000000023015118%22%2C%22uc%22:16}; acw_tc=0a00d69d17752314360228504e90ef8db909830c4039fea4b1b17e107bcab0; websectiga=7750c37de43b7be9de8ed9ff8ea0e576519e8cd2157322eb972ecb429a7735d4; sec_poison_id=b4114ef9-a437-41f5-9fc4-7b9ea74849b6; loadts=1775232324998",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.5615.138 Safari/537.36"
}

# Requsting Data
def post_request(page, url, uri, data):
    a1 = re.search(r'a1=([^;]+)', base_headers['cookie']).group(1)

    signs = sign_with_playwright(
        page=page,
        uri=uri,
        data=data,
        a1=a1,
    )
    sign_header = {
        "X-S": signs["x-s"],
        "X-T": signs["x-t"],
        "x-S-Common": signs["x-s-common"],
        "X-B3-Traceid": signs["x-b3-traceid"],
    }

    headers = {**base_headers, **sign_header}
    data = json.dumps(data, separators=(',', ':'), ensure_ascii=False)
    try:
        response = requests.post(url, headers=headers, data=data.encode('utf-8'))
        response = response.json()
        if response.get('success') is False:
            print("请求失败，cookie 已失效或爬取频繁，请更新 cookie 或更换账号")
            return {"success": False}
        else:
            return {"success": True, "data": response["data"]}
    except Exception as e:
        print(f"请求失败: {e}")
        return {"success": False}

def get_request(page, url, uri, params):
    a1_match = re.search(r'a1=([^;]+)', base_headers['cookie'])
    if not a1_match:
        print("❌ cookie 中没有 a1，直接失败")
        return {"success": False}
    a1 = a1_match.group(1)

    signs = sign_with_playwright(
        page=page,
        uri=uri,
        data=params,
        a1=a1,
    )

    headers = {
        **base_headers,
        "X-S": signs["x-s"],
        "X-T": signs["x-t"],
        "x-S-Common": signs["x-s-common"],
        "X-B3-Traceid": signs["x-b3-traceid"],
    }

    try:
        response = requests.get(url, headers=headers)

        text = response.text.strip()

        if not text.startswith("{"):
            print("❌ 返回不是JSON，被风控了！")
            print("返回内容前100字符：", text[:100])
            return {"success": False}

        data = json.loads(text)

        if data.get('success') is False:
            print("❌ 接口返回失败，cookie可能失效")
            return {"success": False}

        return {"success": True, "data": data.get("data")}

    except Exception as e:
        print("❌ JSON解析失败:", e)
        return {"success": False}

# Change the Time
def get_time(ctime):
    timeArray = time.localtime(int(ctime / 1000))
    return time.strftime("%Y.%m.%d", timeArray)

# Save Data of Post
def sava_data(comment, note_title, note_content, note_url, writer, label):
    global note_comment_count, total_comment_count
    note_comment_count += 1
    total_comment_count += 1

    data_dict = {
        "笔记标题": note_title,
        "笔记内容": note_content,
        "笔记链接": note_url,
        "评论级别": label,
        "用户ID": comment['user_info']['user_id'].strip(),
        "用户名": comment['user_info']['nickname'].strip(),
        "头像链接": comment['user_info']['image'].strip(),
        "评论时间": get_time(comment['create_time']),
        "IP属地": comment.get('ip_location', '未知'),
        "点赞数量": comment['like_count'],
        "评论内容": comment['content'].strip().replace('\n', '')
    }

    print(f"当前笔记评论数: {note_comment_count}, 总评论数：{total_comment_count}\n",
          f"笔记标题：{data_dict['笔记标题']}\n",
          f"笔记内容：{data_dict['笔记内容']}\n",
          f"笔记链接：{data_dict['笔记链接']}\n",
          f"评论级别：{data_dict['评论级别']}\n",
          f"用户名：{data_dict['用户名']}\n",
          f"评论时间：{data_dict['评论时间']}\n",
          f"评论内容：{data_dict['评论内容']}\n"
          )
    writer.writerow(data_dict)

# Main Content of the Post
def get_note_content(page, note_id, xsec_token):
    try:
        note_url = f"https://www.xiaohongshu.com/explore/{note_id}?xsec_token={xsec_token}"

        # 打开页面，等待网络空闲
        response = page.goto(note_url, timeout=60000)
        if not response or response.status != 200:
            print(f"❌ 页面无法访问: {note_url} (状态码: {response.status if response else 'None'})")
            return ""

        # 等待页面 JS 渲染完成
        page.wait_for_load_state("networkidle", timeout=60000)

        # 等待正文元素出现（模糊匹配 class 避免变化）
        locator = page.locator("div[class*='note-content']")
        try:
            locator.wait_for(timeout=60000)
        except TimeoutError:
            print(f"❌ 正文元素未出现，可能被限制或异步加载失败: {note_url}")
            return ""

        # 获取正文内容
        content = locator.inner_text().strip()
        return content

    except Exception as e:
        print(f"❌ 页面抓取正文失败: {e} | 链接: {note_url}")
        return ""

# Subcomments
def get_sub_comments(page, note_id, root_comment_id, sub_comment_cursor, xsec_token, note_title, note_content, note_url, note_comment_max_count, writer):
    while True:
        params = {
            "note_id": note_id,
            "root_comment_id": root_comment_id,
            "cursor": sub_comment_cursor,
            "num": 10,
            "top_comment_id": "",
            "image_scenes": "jpg,webp,avif",
            "xsec_token": xsec_token
        }
        params_str = '&'.join([f"{k}={v}" for k, v in params.items()])
        url = f"https://edith.xiaohongshu.com/api/sns/web/v2/comment/sub/page?{params_str}"
        uri = f"/api/sns/web/v2/comment/sub/page"

        time.sleep(random.uniform(0.5, 3))
        response = get_request(page, url, uri=uri, params=params)
        if response.get('success') is False or response.get('data') is None:
            break

        sub_comment_data = response["data"]
        for sub_comment in sub_comment_data['comments']:
            if note_comment_count >= note_comment_max_count:
                return
            sava_data(sub_comment, note_title, note_content, note_url, writer, label="一级评论")

        if not sub_comment_data['has_more']:
            break
        sub_comment_cursor = sub_comment_data['cursor']

# Comments
def get_comments(page, note_id, xsec_token, note_title, note_content, note_comment_max_count, writer):
    cursor = ''
    page_num = 0
    note_url = f'https://www.xiaohongshu.com/explore/{note_id}?xsec_token={xsec_token}&xsec_source=pc_feed'
    while True:
        params = {
            "note_id": note_id,
            "cursor": cursor,
            "top_comment_id": "",
            "image_scenes": "jpg,webp,avif",
            "xsec_token": xsec_token
        }
        params_str = '&'.join([f"{k}={v}" for k, v in params.items()])
        url = f'https://edith.xiaohongshu.com/api/sns/web/v2/comment/page?{params_str}'
        uri = f"/api/sns/web/v2/comment/page"

        time.sleep(random.uniform(1, 4))
        response = get_request(page, url, uri=uri, params=params)
        if response.get('success') is False or response.get('data') is None:
            return False

        comment_data = response["data"]
        for comment in comment_data['comments']:
            if note_comment_count >= note_comment_max_count:
                return True
            sava_data(comment, note_title, note_content, note_url, writer, label="一级评论")

            # 默认不爬取子评论
            is_sub_comments = False
            if is_sub_comments and len(comment['sub_comments']) != 0:
                sava_data(comment['sub_comments'][0], note_title, note_content, note_url, writer, label="二级评论")
                get_sub_comments(page, note_id, comment['id'], comment['sub_comment_cursor'], xsec_token, note_title, note_content, note_url, note_comment_max_count, writer)

        if not comment_data['has_more']:
            return True
        cursor = comment_data['cursor']
        page_num += 1
        print('================爬取Page{}完毕================'.format(page_num))

# Searching Posts
def keyword_search(page, keyword, note_type, sort_type, filter_note_time, max_page_count, note_comment_max_count, writer):
    note_url = "https://edith.xiaohongshu.com/api/sns/web/v1/search/notes"
    page_num = 1
    while True:
        filters = [
            {"tags": [sort_type_dict[sort_type]], "type": "sort_type"},
            {"tags": ["不限"], "type": "filter_note_type"},
            {"tags": filter_note_time, "type": "filter_note_time"},
            {"tags": ["不限"], "type": "filter_note_range"},
            {"tags": ["不限"], "type": "filter_pos_distance"}
        ]

        data = {
            "ext_flags": [],
            "filters": filters,
            "geo": "",
            "image_formats": ["jpg", "webp", "avif"],
            "keyword": keyword,
            "note_type": note_type_dict[note_type],
            "page": page_num,
            "page_size": 20,
            'search_id': get_search_id(),
            "sort": "general"
        }

        response = post_request(page, note_url, uri='/api/sns/web/v1/search/notes', data=data)
        print("搜索接口返回:", response)
        if response.get('success') is False:
            return

        json_data = response['data']
        notes = json_data['items']
        for note in notes:
            if note['model_type'] != "note":
                continue
            note_id = note['id']
            xsec_token = note['xsec_token']
            note_title = note['note_card'].get('display_title', '')
            note_content = get_note_content(page, note_id, xsec_token)

            print(f"请求成功，正在爬取笔记标题：{note_title}的评论\n")
            global note_comment_count
            note_comment_count = 0
            if not get_comments(page, note_id, xsec_token, note_title, note_content, note_comment_max_count, writer):
                return

        if response['data']['has_more'] is False or page_num >= max_page_count:
            break
        else:
            page_num += 1

# Main
def main():
    print("进入 main 函数")
    keyword = '酷儿认同'  
    note_type = '图文'  
    sort_type = '最多评论'  
    filter_note_time = '不限'  

    header = ["笔记标题", "笔记内容", "笔记链接", "评论级别", "用户ID", "用户名", "头像链接", "评论时间", "IP属地", "点赞数量", "评论内容"]
    f = open(f"{keyword}_{note_type}_{sort_type}_{filter_note_time}.csv", "w", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(f, header)
    writer.writeheader()

    max_page_count = 1
    note_comment_max_count = 500

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        browser_context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=base_headers['user-agent']
        )

        stealth_path = "static/stealth.min.js"
        if os.path.exists(stealth_path):
            browser_context.add_init_script(path=stealth_path)
        else:
            print("未找到 stealth.min.js，已跳过反检测脚本加载")

        print("👉 浏览器已启动")
        context_page = browser_context.new_page()
        print("👉 准备打开小红书")
        context_page.goto("https://www.xiaohongshu.com", timeout=30000)
        print("👉 页面加载完成")
        print("👉 开始执行搜索")
        keyword_search(context_page, keyword, note_type, sort_type, filter_note_time, max_page_count, note_comment_max_count, writer)

main()