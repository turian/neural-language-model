#!/usr/bin/python
#
#  For every filename in sys.stdin, add a file BAD to that run directory.
#  Read stdin until there is a blank line.
#
#   BUG: If the filename has a space in it, sorry you're out of luck.
#   BUG: We don't unescape quotes, we just strip them.
#

import sys, os.path, string
#assert len(sys.argv)>2

while 1:
#    for l in sys.stdin:
    l = sys.stdin.readline()
#    for l in sys.stdin:
    if string.strip(l) == "": break
    for f in string.split(l):
        f = f.replace('\"','').replace("\'",'')
        if not os.path.exists(f): continue
        d = os.path.dirname(os.path.realpath(f))
        newf = os.path.join(d, "BAD")
        print newf
        if os.path.exists(newf): continue
        cmd = "rm %s" % os.path.join(d, "*.dat")
        print >> sys.stderr, "Creating %s, %s" % (newf, cmd)
        open(newf, "wt").close()
        os.system("rm %s" % os.path.join(d, "*.dat"))
