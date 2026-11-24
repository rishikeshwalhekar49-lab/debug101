# Snippets

Fragments of training code, the kind pasted from a notebook or a chat, for `snippet`:

    python -m silent_bug_agent snippet examples/snippets/training_step.txt
    pbpaste | python -m silent_bug_agent snippet            # macOS: check what is on the clipboard
    Get-Clipboard | python -m silent_bug_agent snippet      # Windows PowerShell

A snippet is only read, never run, so its findings are not validated. To test a
repair, run `fix` on the full training script.
