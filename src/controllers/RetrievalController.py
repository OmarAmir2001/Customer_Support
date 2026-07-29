from .BaseController import BaseController


class RetrievalController(BaseController):
    def __init__(self):
        super().__init__()

    def retrierve_question(self, question: str, department: str) -> list:
        """
        Retrieves the chunks from the database based on the question and department.
        """
        return ['chunk1', 'chunk2', 'chunk3']
