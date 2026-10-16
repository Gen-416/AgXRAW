# SPDX-License-Identifier: GPL-3.0-or-later
"""Application development entry; normal RAW batch CLI stays unchanged."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='AgXRAW 共享应用服务')
    parser.add_argument('--serve', action='store_true', help='启动本机接口')
    parser.add_argument('--handshake', action='store_true', help='向父进程报告就绪信息')
    parser.add_argument('--frontend', type=Path, help='已构建的独立前端目录')
    parser.add_argument('--open', action='store_true', help='在浏览器打开独立前端')
    parser.add_argument('--schema', action='store_true', help='输出版本化接口合同')
    parser.add_argument('--request', type=Path, help='执行一份 JSON 应用请求；- 表示标准输入')
    args = parser.parse_args()
    if args.schema:
        import json
        from .contracts import schema
        print(json.dumps(schema(), ensure_ascii=False, indent=2))
        return 0
    if args.request is not None:
        import json
        import sys
        from contextlib import redirect_stdout
        from .api import Application
        try:
            limit = 96 * 1024 * 1024
            if str(args.request) == '-':
                data = sys.stdin.buffer.read(limit + 1)
            else:
                with args.request.open('rb') as stream:
                    data = stream.read(limit + 1)
            if len(data) > limit:
                raise ValueError('请求过大')
            with redirect_stdout(sys.stderr):
                result = Application().execute(json.loads(data))
        except (OSError, ValueError) as exc:
            result = Application._error(None, 'invalid-request', str(exc))
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result['ok'] else 1
    if not args.serve:
        parser.error('请选择 --serve、--schema 或 --request')
    from dngscan.transports.server import run
    return run(frontend_dir=args.frontend, open_browser=args.open, handshake=args.handshake)


if __name__ == '__main__':
    raise SystemExit(main())
