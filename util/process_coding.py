import chardet


def process_coding(path):
    """
    检测txt文档的编码格式，根据编码格式打开
    """
    with open(path, 'rb') as fp:
        file_data = fp.read()
        coding_result = chardet.detect(file_data)
        file_content = file_data.decode(encoding=coding_result['encoding']).strip()
    return file_content
