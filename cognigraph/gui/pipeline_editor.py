"""Pipeline editor for cognigraph"""

if __name__ == '__main__':
    from PyQt5.QtWidgets import QMainApplication
    import sys
    app = QApplication(sys.argv)
    dialog = PipelineEditorDialog()
    dialog.show()
    sys.exit(app.exec_())
