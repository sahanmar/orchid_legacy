echo "Mypy check..."
rm -rf .mypy_cache
mypy . --exclude data || exit $?