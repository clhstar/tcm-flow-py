from langchain.tools import tool


@tool("present_files")
def present_files(filepaths: list[str]) -> str:
    """
    展示已经生成的文件路径
    第一版只返回路径，不做真正的文件下载

    Args:
        filepaths: 文件路径列表

    Returns:
        格式化的文件路径字符串
    """
    return "已生成文件：" + ", ".join(filepaths)