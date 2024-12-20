from toolbox import update_ui
from toolbox import CatchException, report_exception

# 从toolbox模块中导入write_history_to_file和promote_file_to_downloadzone函数
from toolbox import write_history_to_file, promote_file_to_downloadzone
from crazy_functions.crazy_utils import request_gpt_model_in_new_thread_with_ui_alive


def 解析Paper(
    file_manifest,
    project_folder,
    llm_kwargs,
    plugin_kwargs,
    chatbot,
    history,
    system_prompt,
):
    import time
    import glob
    import os

    """ glob模块提供了一个函数glob()，用于在目录中查找符合特定规则的文件路径名，支持通配符。 """
    for index, fp in enumerate(file_manifest):
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            file_content = f.read()

        prefix = "接下来请你逐文件分析下面的论文文件，概括其内容" if index == 0 else ""
        i_say = (
            prefix
            + f"请对下面的文章片段用中文做一个概述，文件名是{os.path.relpath(fp, project_folder)}，文章内容是 ```{file_content}```"
        )
        i_say_show_user = (
            prefix
            + f"[{index+1}/{len(file_manifest)}] 请对下面的文章片段做一个概述: {os.path.abspath(fp)}"
        )
        chatbot.append((i_say_show_user, "[Local Message] waiting gpt response."))
        yield from update_ui(chatbot=chatbot, history=history)  # 刷新界面

        msg = "正常"
        gpt_say = yield from request_gpt_model_in_new_thread_with_ui_alive(
            i_say,
            i_say_show_user,
            llm_kwargs,
            chatbot,
            history=[],
            sys_prompt=system_prompt,
        )  # 带超时倒计时
        chatbot[-1] = (i_say_show_user, gpt_say)
        history.append(i_say_show_user)
        history.append(gpt_say)
        yield from update_ui(chatbot=chatbot, history=history, msg=msg)  # 刷新界面
        time.sleep(2)

    all_file = ", ".join(
        [os.path.relpath(fp, project_folder) for index, fp in enumerate(file_manifest)]
    )
    i_say = f"根据以上你自己的分析，对全文进行概括，用学术性语言写一段中文摘要，然后再写一段英文摘要（包括{all_file}）。"
    chatbot.append((i_say, "[Local Message] waiting gpt response."))
    yield from update_ui(chatbot=chatbot, history=history)  # 刷新界面

    msg = "正常"
    # ** gpt request **
    gpt_say = yield from request_gpt_model_in_new_thread_with_ui_alive(
        i_say, i_say, llm_kwargs, chatbot, history=history, sys_prompt=system_prompt
    )  # 带超时倒计时

    chatbot[-1] = (i_say, gpt_say)
    history.append(i_say)
    history.append(gpt_say)
    yield from update_ui(chatbot=chatbot, history=history, msg=msg)  # 刷新界面
    res = write_history_to_file(history)
    promote_file_to_downloadzone(res, chatbot=chatbot)
    chatbot.append(("完成了吗？", res))
    yield from update_ui(chatbot=chatbot, history=history, msg=msg)  # 刷新界面


@CatchException
def 读文章写摘要(
    txt, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, user_request
):
    history = []  # 清空历史，以免输入溢出
    import glob, os

    if os.path.exists(txt):
        project_folder = txt
    else:
        if txt == "":
            txt = "空空如也的输入栏"
        report_exception(
            chatbot, history, a=f"解析项目: {txt}", b=f"找不到本地项目或无权访问: {txt}"
        )
        yield from update_ui(chatbot=chatbot, history=history)  # 刷新界面
        return
    file_manifest = [
        f for f in glob.glob(f"{project_folder}/**/*.tex", recursive=True)
    ]  # + \
    # [f for f in glob.glob(f'{project_folder}/**/*.cpp', recursive=True)] + \
    # [f for f in glob.glob(f'{project_folder}/**/*.c', recursive=True)]
    if len(file_manifest) == 0:
        report_exception(
            chatbot, history, a=f"解析项目: {txt}", b=f"找不到任何.tex文件: {txt}"
        )
        yield from update_ui(chatbot=chatbot, history=history)  # 刷新界面
        return
    yield from 解析Paper(
        file_manifest,
        project_folder,
        llm_kwargs,
        plugin_kwargs,
        chatbot,
        history,
        system_prompt,
    )
