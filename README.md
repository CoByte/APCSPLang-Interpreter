# APCSPLang-Interpreter

An interpreter for the APCSP psuedocode language used on the AP test.

## Features

In theory, this interpreter should conform to the APCSP language specification avaliable [here](https://apcentral.collegeboard.org/pdf/ap-computer-science-principles-exam-reference-sheet.pdf). With a few notable exceptions, any code from the test should be able to be copy-pasted directly into the **code.apcsp** file, and then run to produce a result. However, it is important to note that though things that have been specified to work will work, and things that have been explicitly specified to not work will not work, there is a significant grey area where code that one might not expect to work will work. Many of those things are outlined below.

### Weird Stuff

* The spec uses characters that cannot be easily typed for some boolean operators, these being:

  * ≠
  * ≥
  * ≤
  
  To make things more typable, these more typable symbols have been added as equivalent:
  
  * /=
  * \>=
  * <=

  Note that the weird symbols do still work, so they won't need to be changed if copy-pasting.
  
* **TRUE** and **FALSE** keywords have been added
* Procedures that do not have a return statement will return the **NONE** type
* Procedures are stored in memory as variables, and thus can be passed into functions. Note that procedures do not capture local scope like they might in a better language, so I don't think decorator patterns and similar things are possible.
* Strings can be written with quotations like in most languages, e.g *"I'm a string!"*
  * Note that no operators apply to strings except for equals/nequals, so concatenation is not possible.

## The Robot

In order to use the robot, the board must first be setup through the use of a few custom procedures. They are:

* **SETUP**: Takes two integers as arguments, x and y. This will be the size of the board. This MUST be called before the robot is used.
* **CLOSE_BOX**: Takes two integers as arguments, x and y. It will fill a square on the board, and make it not passable.
* **CLOSE_MULTI_BOX**: Takes two lists of integers as arguments. The first list is all the x coords, and the second list is all the corresponding y coords. All of the supplied coords will be filled, and will no longer be passable.

There's also some other weirdness with the robot. It will always start facing left in the bottom right corner. If it needs to be in a different place, you'll have to move the robot yourself.

The functionality of the CAN_MOVE procedure is also slightly odd. The procedure is designed to take in a string argument, as the spec is unclear as to how the arguments should be passed. However, I have seen examples where the argument uses the keywords **forwards**, **backwards**, **left**, and **right** instead of strings. To alleviate this issue, 4 variables that map from the keyword to the string have been included, so that syntax remains copy-pastable.




