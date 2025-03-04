#!/usr/bin/env python3
"""
Note
----

This file is copied (possibly with major modifications) from the
sources of the scipy project - https://github.com/scipy/scipy.
It remains licensed as the rest of scipy (BSD-3 license as of October 2023).

# ## ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the scipy package for the
#   copyright and license terms.
#
# ## ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
import os
import textwrap


def init_version():
    init = os.path.join(os.path.dirname(__file__), '../pyproject.toml')
    with open(init) as fid:
        data = fid.readlines()

    version_line = next(
        line for line in data if line.startswith('version =')
    )

    version = version_line.strip().split(' = ')[1]
    version = version.replace('"', '').replace("'", '')

    return version


def git_version(version):
    # Append last commit date and hash to dev version information,
    # if available

    import subprocess
    import os.path

    git_hash = ''
    try:
        p = subprocess.Popen(
            ['git', 'log', '-1', '--format="%H %aI"'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(__file__),
        )
    except FileNotFoundError:
        pass
    else:
        out, err = p.communicate()
        if p.returncode == 0:
            git_hash, git_date = (
                out.decode('utf-8')
                .strip()
                .replace('"', '')
                .split('T')[0]
                .replace('-', '')
                .split()
            )

            # Only attach git tag to development versions
            if 'dev' in version:
                version += f'+git{git_date}.{git_hash[:7]}'

    return version, git_hash


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--write', help="Save version to this file")
    parser.add_argument(
        '--meson-dist',
        help='Output path is relative to MESON_DIST_ROOT',
        action='store_true'
    )
    args = parser.parse_args()

    version, git_hash = git_version(init_version())

    template = textwrap.dedent(f'''
        """
        Module to expose more detailed version info for the installed `scipy`
        """
        version = "{version}"
        full_version = version
        short_version = version.split('.dev')[0]
        git_revision = "{git_hash}"
        release = 'dev' not in version and '+' not in version

        if not release:
            version = full_version
    ''')

    if args.write:
        outfile = args.write
        if args.meson_dist:
            outfile = os.path.join(
                os.environ.get('MESON_DIST_ROOT', ''),
                outfile
            )

        # Print human readable output path
        relpath = os.path.relpath(outfile)
        if relpath.startswith('.'):
            relpath = outfile

        with open(outfile, 'w') as f:
            f.write(template)
    else:
        print(version)
