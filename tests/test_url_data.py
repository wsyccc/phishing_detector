import pytest

from src.url_data import URLData


def test_long_url_length():
    """
    1. Correctly calculate and return the URL length feature.
       https://very-long-url-example.com/with/a/very/long/path
       验证是否能正确计算 URL 长度
    """
    url = "https://very-long-url-example.com/with/a/very/long/path"
    data = URLData(url)
    features = data.extract_features()
    # 预期长度可以自行计算或让测试灵活一点，比如只要长度大于某值即可
    assert features["url_length"] == len(url)
    # 也可以写成：
    # assert features["url_length"] > 50  # 示例：确保足够长
    assert data.is_valid is True


def test_https_detection():
    """
    2. Identify HTTPS-enabled and non-HTTPS-enabled websites.
       这里演示同时测试 https://secure.com 和 http://insecure.com 两种情况
       - https://secure.com => is_https == True
       - http://insecure.com => is_https == False
    """
    secure_url = "https://secure.com"
    insecure_url = "http://insecure.com"

    secure_data = URLData(secure_url)
    insecure_data = URLData(insecure_url)

    assert secure_data.extract_features()["is_https"] is True, "应该识别为HTTPS"
    assert insecure_data.extract_features()["is_https"] is False, "应该识别为非HTTPS"

    # 也可以进一步检查 has_all_features 是否正确
    assert secure_data.has_all_features is True
    assert insecure_data.has_all_features is True


def test_ipv4_detection():
    """
    3. Identifies URLs that use IP addresses.
       http://192.168.1.1/test => is_ipv4 == True
    """
    ip_url = "http://192.168.1.1/test"
    data = URLData(ip_url)
    features = data.extract_features()

    assert features["is_ipv4"] is True, "应该识别为IP地址"
    # 验证解析是否有效
    assert data.is_valid is True


def test_url_with_redirect():
    """
    4. URLs with redirects:
       http://example.com/redirect?to=phishingsite.com
       The system recognizes the redirection and returns the corresponding feature values
       由于当前 URLData 类并没有特殊的“重定向”检测逻辑，这里主要测试提取功能。
    """
    redirect_url = "http://example.com/redirect?to=phishingsite.com"
    data = URLData(redirect_url)
    features = data.extract_features()

    # 验证是否能正常解析出 hostname, length 等
    assert data.is_valid is True
    assert features["ngram_hostname"] == ['exa', 'xam', 'amp', 'mpl', 'ple', 'le.', 'e.c', '.co', 'com']
    assert features["url_length"] == len(redirect_url)

    # 检查问号数量，判断是否正确计数
    # => 该 URL 有 1 个问号
    assert features["number_of_question_marks"] == 1

    # 你也可以根据需求去检查有没有一些“重定向”字段等
    # 比如这个 URL 里带有 "redirect"，也可以用来判定某种模式。
    assert "redirect" in data.parsed.path


@pytest.mark.parametrize(
    "test_url,expected_port",
    [
        ("http://example.com", None),
        ("http://example.com:8080/path", 8080),
        ("https://secure.com:443", 443),
    ]
)
def test_url_port_detection(test_url, expected_port):
    """
    附加示例：测试带端口或不带端口的 URL 是否能正确提取 port
    """
    data = URLData(test_url)
    features = data.extract_features()
    assert features["has_port"] == expected_port
