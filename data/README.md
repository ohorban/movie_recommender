# data/

Put your unzipped Letterboxd export folder here:

```
data/letterboxd-<username>-<YYYY-MM-DD-HH-MM>-utc/
```

Export it from <https://letterboxd.com/settings/data>. The newest folder is always the one used, so
old exports can be left in place. Export folders themselves are git-ignored — see `.gitignore`.

Everything else that lands in here (`cache/`, `external/`) is downloaded automatically and is
rebuildable, so it is ignored too.
