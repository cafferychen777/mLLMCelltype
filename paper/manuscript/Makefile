# LaTeX编译Makefile
# 使用方法：
#   make         - 编译PDF
#   make clean   - 清理辅助文件
#   make watch   - 启动自动监控编译
#   make force   - 强制重新编译（清理后编译）

TEX_FILE = sn-article.tex
BASE_NAME = sn-article
PDF_FILE = $(BASE_NAME).pdf

# 辅助文件列表
AUX_FILES = $(BASE_NAME).aux $(BASE_NAME).log $(BASE_NAME).bbl $(BASE_NAME).blg \
           $(BASE_NAME).out $(BASE_NAME).toc $(BASE_NAME).lof $(BASE_NAME).lot \
           $(BASE_NAME).fdb_latexmk $(BASE_NAME).fls $(BASE_NAME).synctex.gz

.PHONY: all clean watch force help

# 默认目标
all: $(PDF_FILE)

# 编译PDF
$(PDF_FILE): $(TEX_FILE) sn-bibliography.bib
	@echo "开始编译 $(TEX_FILE)..."
	@pdflatex -interaction=nonstopmode $(TEX_FILE)
	@if [ -f sn-bibliography.bib ]; then \
		echo "运行 BibTeX..."; \
		bibtex $(BASE_NAME); \
		echo "第二次 pdflatex 编译..."; \
		pdflatex -interaction=nonstopmode $(TEX_FILE); \
	fi
	@echo "第三次 pdflatex 编译..."
	@pdflatex -interaction=nonstopmode $(TEX_FILE)
	@echo "✅ 编译完成：$(PDF_FILE)"

# 清理辅助文件
clean:
	@echo "清理辅助文件..."
	@rm -f $(AUX_FILES)
	@echo "✅ 清理完成"

# 清理所有文件（包括PDF）
cleanall: clean
	@rm -f $(PDF_FILE)
	@echo "✅ 清理所有文件完成"

# 强制重新编译
force: clean all

# 启动自动监控编译（需要 latexmk）
watch:
	@echo "启动自动监控编译..."
	latexmk -pdf -pvc -pdflatex="pdflatex -interaction=nonstopmode" $(TEX_FILE)

# 显示帮助信息
help:
	@echo "LaTeX编译工具使用方法："
	@echo "  make         - 编译PDF"
	@echo "  make clean   - 清理辅助文件"
	@echo "  make cleanall- 清理所有文件（包括PDF）"
	@echo "  make watch   - 启动自动监控编译"
	@echo "  make force   - 强制重新编译"
	@echo "  make help    - 显示此帮助信息"