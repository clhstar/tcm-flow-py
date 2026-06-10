from langchain.tools import tool


@tool("present_files")
def present_files(filepaths: list[str]) -> str:
    """
    展示已经生成的文件路径。

    当前 V0.9 只是架构保留。
    后续可用于生成问诊报告、导出 PDF、导出 Word 等。
    """
    return "已生成文件：" + ", ".join(filepaths)