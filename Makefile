
# ===== Auto-generated push target =====
# Usage: make submit MSG="your message"
MSG ?= auto commit
submit:
	@if [ -z "$(MSG)" ]; then \
		echo "Error: missing commit message. Use: make submit MSG=\"xxx\""; \
		exit 1; \
	fi
	git add -A
	git commit -m "$(MSG)" || echo "No changes to commit"
	git push origin HEAD
	@echo "Push complete"
