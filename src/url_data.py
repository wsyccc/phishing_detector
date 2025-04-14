from urllib.parse import urlparse
from difflib import SequenceMatcher
from collections import Counter

class URLData:
    def __init__(self, url, label=None):
        # If the URL does not start with "http://" or "https://", prepend "http://"
        if not (url.startswith("http://") or url.startswith("https://")):
            url = "http://" + url
        self.url = url
        # Attempt to parse the URL and catch any parsing exceptions
        try:
            self.parsed = urlparse(url)
        except ValueError:
            self.parsed = None
        self.is_valid = self._validate_url()
        self.label = label
        self.has_all_features = self._has_all_features()

    def _validate_url(self):
        # If parsing failed, return False immediately
        if self.parsed is None:
            return False
        if not self.parsed.scheme or not self.parsed.netloc:
            return False
        return True

    def _has_all_features(self):
        if self.is_valid:
            features = self.extract_features()
            # Note: This loop only checks the first feature; consider checking all features instead.
            for key, value in features.items():
                # If any feature is None, return False
                if value is None:
                    return False
            return True
        return False

    def parsed_url(self):
        try:
            return urlparse(self.url)
        except ValueError:
            return None

    def url_length(self):
        try:
            return len(self.url) if len(self.url) > 0 else None
        except ValueError:
            return None

    def hostname(self):
        try:
            return self.parsed.netloc if self.parsed and self.parsed.netloc else None
        except ValueError:
            return None

    def ngram_hostname(self, n=3):
        try:
            host = self.hostname()
            return [host[i:i+n] for i in range(len(host) - n + 1)] if host else None
        except ValueError:
            return None

    def hostname_length(self):
        try:
            host = self.hostname()
            return len(host) if host and len(host) > 0 else None
        except ValueError:
            return None

    def isIpv4(self):
        try:
            host = self.hostname()
            return host.replace('.', '').isnumeric() if host else None
        except ValueError:
            return None

    def TLD(self):
        try:
            # If there is no dot in the path, return an empty string
            parts = self.parsed.path.split('.') if self.parsed else []
            return "tld_" + parts[-1] if parts and len(parts[-1]) > 0 else None
        except ValueError:
            return None

    def isHTTPS(self):
        try:
            return self.parsed.scheme == 'https' if self.parsed else False
        except ValueError:
            return None

    def number_of_dots(self):
        try:
            return self.url.count('.')
        except ValueError:
            return None

    def number_of_letters(self):
        try:
            return sum(c.isalpha() for c in self.url)
        except ValueError:
            return None

    def letter_ratio(self):
        try:
            return self.number_of_letters() / self.url_length() if self.url_length() else None
        except ValueError:
            return None

    def number_of_digits(self):
        try:
            return sum(c.isdigit() for c in self.url)
        except ValueError:
            return None

    def digit_ratio(self):
        try:
            return self.number_of_digits() / self.url_length() if self.url_length() else None
        except ValueError:
            return None

    def number_of_equals_in_url(self):
        try:
            return self.url.count('=')
        except ValueError:
            return None

    def number_of_question_marks(self):
        try:
            return self.url.count('?')
        except ValueError:
            return None

    def number_of_ampersands(self):
        try:
            return self.url.count('&')
        except ValueError:
            return None

    def is_punycode(self):
        try:
            host = self.hostname()
            return host.startswith('xn--') if host else False
        except ValueError:
            return None

    def has_port(self):
        try:
            return self.parsed.port if self.parsed else None
        except ValueError:
            return None

    def number_of_hyphens(self):
        try:
            return self.url.count('-')
        except ValueError:
            return None

    def char_repeat(self):
        try:
            char_counts = Counter(self.url)
            total_chars = len(self.url)
            repeated_chars = sum(count for count in char_counts.values() if count > 1)
            return repeated_chars / total_chars if total_chars else None
        except ValueError:
            return None

    def number_of_special_characters(self):
        try:
            return sum(not c.isalnum() for c in self.url)
        except ValueError:
            return None

    def special_character_ratio(self):
        try:
            return self.number_of_special_characters() / self.url_length() if self.url_length() else None
        except ValueError:
            return None

    def url_similarity_index(self):
        try:
            host = self.hostname()
            return SequenceMatcher(None, self.url, host).ratio() * 100 if host else None
        except ValueError:
            return None

    def char_continuation_rate(self):
        try:
            count = 0
            for i in range(1, len(self.url)):
                if self.url[i].isalpha() and self.url[i - 1].isalpha():
                    count += 1
            return count / len(self.url) if len(self.url) else None
        except ValueError:
            return None

    def tld_legitimate_prob(self):
        try:
            tld_legitimate_probs = {
                'com': 0.9, 'org': 0.85, 'net': 0.8, 'info': 0.3, 'xyz': 0.2
            }
            # Extract the actual part of the tld string, for example "tld_com" -> "com"
            tld = self.TLD()
            if tld:
                key = tld.split('_')[-1]
                return tld_legitimate_probs.get(key, 0.1)
            else:
                return 0.1
        except ValueError:
            return None

    def has_obfuscation(self):
        try:
            return any(x in self.url for x in ['%', '#', '@', '!', '~', '$', '*', '_', '(', ')', '[', ']', '{', '}', '|', '\\', '^', '<', '>', '`', ';', ':', '/', ','])
        except ValueError:
            return None

    def number_of_obfuscation(self):
        try:
            return sum(x in self.url for x in ['%', '#', '@', '!', '~', '$', '*', '_', '(', ')', '[', ']', '{', '}', '|', '\\', '^', '<', '>', '`', ';', ':', '/', ','])
        except ValueError:
            return None

    def obfuscation_ratio(self):
        try:
            return self.number_of_obfuscation() / self.url_length() if self.url_length() else None
        except ValueError:
            return None

    def extract_features(self):
        return {
            'url': self.url,
            'url_length': self.url_length(),
            'ngram_hostname': self.ngram_hostname(),
            'hostname_length': self.hostname_length(),
            'is_ipv4': self.isIpv4(),
            'tld': self.TLD(),
            'is_https': self.isHTTPS(),
            'number_of_dots': self.number_of_dots(),
            'number_of_letters': self.number_of_letters(),
            'letter_ratio': self.letter_ratio(),
            'number_of_digits': self.number_of_digits(),
            'digit_ratio': self.digit_ratio(),
            'number_of_equals_in_url': self.number_of_equals_in_url(),
            'number_of_question_marks': self.number_of_question_marks(),
            'number_of_ampersands': self.number_of_ampersands(),
            'is_punycode': self.is_punycode(),
            'has_port': self.has_port(),
            'number_of_hyphens': self.number_of_hyphens(),
            'char_repeat': self.char_repeat(),
            'number_of_special_characters': self.number_of_special_characters(),
            'special_character_ratio': self.special_character_ratio(),
            'url_similarity_index': self.url_similarity_index(),
            'char_continuation_rate': self.char_continuation_rate(),
            'tld_legitimate_prob': self.tld_legitimate_prob(),
            'has_obfuscation': self.has_obfuscation(),
            'number_of_obfuscation': self.number_of_obfuscation(),
            'obfuscation_ratio': self.obfuscation_ratio(),
            'label': self.label
        }
