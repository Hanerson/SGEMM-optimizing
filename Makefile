MSG ?= auto commit
submit:
	@if [ -z "$(MSG)" ]; then \
		echo "错误：请指定提交信息，如 make submit MSG=\"xxx\""; \
		exit 1; \
	fi
	git add -A
	git commit -m "$(MSG)" || echo "没有需要提交的更改"
	git push origin HEAD
	@echo "✅ 推送完成"
