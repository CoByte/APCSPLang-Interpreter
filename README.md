# WORK IN PROGRESS

# APCSPLang-Interpreter

An interpreter for the APCSP psuedocode language used on the AP test.

## Features

In theory, this interpreter should conform to the APCSP language specification avaliable [here](https://apcentral.collegeboard.org/pdf/ap-computer-science-principles-exam-reference-sheet.pdf). With a few notable exceptions, any code from the test should be able to be copy-pasted directly into the **code.apcsp** file, and then run to produce a result.

## Notable exceptions

Because of the weirdness of the language spec, there are some things that are not directly 1 to 1 compatible, or require some setup. They are as follows:

* The spec uses characters that cannot be easily typed for some boolean operators, these being:

  * ≠
  * ≥
  * ≤
  
  To make things more typable, these characters have been replaced with the more traditional symbols:
  
  * /=
  * >=
  * <=
