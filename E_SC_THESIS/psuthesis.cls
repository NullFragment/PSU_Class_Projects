%%% ======================================================================
%%%  @LaTeX-file{
%%%     filename  = "psuthesis.cls",
%%%     version   = "2.4.0",
%%%     date      = "2013/10/21",
%%%     time      = "16:00:00 EDT",
%%%     author    = "Gary L. Gray",
%%%     copyright = "Gary L. Gray",
%%%     address   = "Engineering Science and Mechanics,
%%%                  212 Earth & Engineering Sciences Bldg.,
%%%                  Penn State University,
%%%                  University Park, PA 16802,
%%%                  USA",
%%%     telephone = "814-863-1778",
%%%     email     = "gray@psu.edu",
%%%     keywords  = "latex, psuthesis, thesis class",
%%%     supported = "yes",
%%%     abstract  = "This package provides a style for typesetting
%%%                  Penn State theses at the bachelors, masters,
%%%                  or Ph.D. level."
%%%  }
%%% ======================================================================
% Change History
%
% Version 2.4.0: **	Given that the Graduate School now allows essentially
%					any line spacing, I have moved the line space setting
%					from psuthesis.cls to this driver file. Go ahead and
%					make it ugly if you want. :-)
%
%				**	Removed \psusigpage since the Graduate School now
%					provides the signature page.
%
%				**	Added the command \collegesubmittedto to add the
%					College in which the thesis/dissertation has been
%					completed to the title page.
%
%				**	Now load the package textcase to deal with the
%					capitalization of the \documenttype on the committee
%					page.
%
% Version 2.3.2: * Changed the implementation of the tocloft package
%                  to remove the ``black boxes'' about which we got so
%                  many queries. It still isn't perfect, but it is better
%                  than it was before. Note that chapter titles that take
%                  two lines or more in the TOC now have hanging indents.
%                  We haven't been able to eliminate this, but we kind of
%                  like it, so we are leaving it as is (for now).
%
% Version 2.3.1: * Added 10pt and 11pt options to the document class,
%                  though we have no idea why anyone would want to use
%                  such insanely small font sizes since it will lead to
%                  line lengths that are much too long.
%
% Version 2.3.0: * Made substantial changes to the psuthesis.cls file
%                  since the draft option was not being passed to the book
%                  class. The change we made also prevents users from
%                  loading options that are not ``recommended''. We
%                  want to thank Jared Craig Ahern for pointing out this
%                  problem and suggesting the fix.
%
% Version 2.2.0: * Two additional class options have been added to
%                  support honors theses submitted to the Schreyer
%                  Honors College. These options are:
%                     - honors
%                     - honorsdepthead
%
%                * We have also added the commands:
%                     - honorsdegreeinfo
%                     - honorsadviser
%                     - honorsdepthead
%
%                  See the driver file for details on these options and
%                  commands.
%
% Version 2.1.2: * Added a note both here and in the template file for
%                  users wanted to implement the subfigure package. That
%                  is, if you want to use the subfigure package you need
%                  to add an option when the tocloft package is loaded.
%
% Version 2.1.1: * Fixed a bug, introduced in 2.1.0, that prevented any
%                  entries from appearing the List of Figures. :-)
%
% Version 2.1.0: * Class now requires the tocloft package to format the
%                  TOC to correspond to PSU Grad School requirements.
%                * Added the \Appendix command to work around a quirk in
%                  LaTeX that makes \addtocontents not work well with
%                  \include.
%                * Eliminated a ton of booleans associated with the
%                  management of the frontmatter. This makes the
%                  frontmatter more intuitive, but it also presents you
%                  with the noose with which you can hang yourself. Sorry.
%                * Added the inlinechaptertoc option to the class. Allows
%                  for an alternate TOC format for Chapter and Appendix
%                  entries.
%                * There is no longer a sigpage option for the class.
%
% Version 2.0.5: * Updated the signature page and added a committee page
%                  to conform to the new grad school requirement.
%                * There is now a sigpage option for the signature page
%                  (it can be turned off) and the standard
%                  now does not have signatures.
%                * Fixed problem with TOC page numbers of List of Figures,
%                  List of Tables, etc.
%                * Cleaned up a few things, especially the boolean logic
%                  for the degree type.
%
%                Version 2.0.5 owes a debt to Penn State graduate student
%                Nils Krahnstoever who sent us a list of bugs along with
%                some excellent suggestions.
%
% Version 2.0.4: Changed the alignment of "Date of Signature" text
%
% Version 2.0.3: Fixed the invocation of \psu@title to allow for \\
%                line breaks. The \MakeUppercase command originally
%                was not, for some reason, working. Unfortunately, we
%                had to resort to some TeX primitives.
%
% Version 2.0.2: Added vita to bs and ba option for the Schreyer's
%                Honors College.
%
% Version 2.0.1: * No longer include the abstract in the TOC.
%                * Double space the title on the title page.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\NeedsTeXFormat{LaTeX2e}[1995/06/01]
\ProvidesClass{psuthesis}[2013/10/18 v2.4.0 psuthesis class]
\RequirePackage{ifthen}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Declare options for different degree types.
% Allowable degrees are:
%     B.S. using class option <bs>
%     Bachelors degree with honors using class option <honors>
%
% The option of an option sets the boolean for that option to
% true and all others to false.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newboolean{psu@secondthesissupervisor}
\newboolean{psu@honorsdepthead}
\newboolean{psu@honors}
\newboolean{psu@bs}
\newboolean{psu@toc}
\newboolean{psu@signature}

\setboolean{psu@secondthesissupervisor}{false}
\setboolean{psu@honorsdepthead}{false}
\setboolean{psu@honors}{false}
\setboolean{psu@bs}{false}
\setboolean{psu@signature}{false}

\DeclareOption{bs}{\setboolean{psu@bs}{true}\setboolean{psu@honors}{false}}
\DeclareOption{inlinechaptertoc}{\setboolean{psu@toc}{true}}
\DeclareOption{honorsdepthead}{\setboolean{psu@honorsdepthead}{true}}
\DeclareOption{secondthesissupervisor}{\setboolean{psu@secondthesissupervisor}{true}}
\DeclareOption{honors}{\setboolean{psu@honors}{true}\setboolean{psu@bs}{false}}%...
\DeclareOption{signature}{\setboolean{psu@signature}{true}}

\DeclareOption{draft}{\PassOptionsToClass{\CurrentOption}{book}}
\DeclareOption{10pt}{\PassOptionsToClass{\CurrentOption}{book}}
\DeclareOption{11pt}{\PassOptionsToClass{\CurrentOption}{book}}
\DeclareOption{12pt}{\PassOptionsToClass{\CurrentOption}{book}}
\DeclareOption*{\PackageWarning{psuthesis}{Unknown option `\CurrentOption'. Ignoring}}
\ExecuteOptions{honors} % the default option is <honors>
\ProcessOptions
\LoadClass[openany,oneside]{book}
\RequirePackage{calc}
\RequirePackage{setspace}
% If you are using the subfigure package, load the tocloft package with
% the subfigure option and comment out the next line.
\RequirePackage{tocloft}[2003/09/26]
%\RequirePackage[subfigure]{tocloft}[2003/09/26]
\RequirePackage{fancyhdr}
\RequirePackage[overload]{textcase}

% Define the margins and other spacings.
\setlength{\oddsidemargin}{0.6in}
\setlength{\topmargin}{0pt}
\setlength{\headheight}{12pt}
\setlength{\headsep}{1.8\baselineskip}
\setlength{\textheight}{8.60in}
\setlength{\textwidth}{5.8in}


%%%%%%%%%%%%%%%%%%%%%%%%
% Settings for tocloft %
%%%%%%%%%%%%%%%%%%%%%%%%
% Format chapter entries so that the chapter name goes on a line
% following "Chapter #".
\renewcommand{\@pnumwidth}{1.75em} % remove TOC margin errors
\renewcommand{\@tocrmarg}{2.75em}
\newlength{\mylength}% a "scratch" length
\newlength{\mylonglength}% another "scratch" length
\ifthenelse{\boolean{psu@toc}}
{%
% Format chapter entries so that the chapter name goes on the same line
% as "Chapter #".
\renewcommand{\cftchappresnum}{Chapter }
\settowidth{\mylength}{\bfseries\cftchappresnum\cftchapaftersnum}% extra space
\addtolength{\cftchapnumwidth}{\mylength} % add the extra space
%
\newcommand{\mylongname}{Appendix }% the longest chapter number header
\settowidth{\mylonglength}{\bfseries\mylongname\cftchapaftersnum}% extra space
}
{%
\renewcommand{\cftchappresnum}{Chapter } 
\renewcommand{\cftchapaftersnumb}{\\ \mbox{}\hspace{-\mylength}\hspace{-0.1em}}
\settowidth{\mylength}{\bfseries\cftchappresnum\cftchapaftersnum} % extra space 
\addtolength{\cftchapnumwidth}{\mylength+0.1em} % add the extra space\renewcommand{\cftchapfont}{\bfseries} 
}%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Here we define internal "commands" that will be used to store the
% thesis title, author name, department, etc.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%Store the Psu-ID Number
\newcommand{\psu@id}{fill}
\providecommand{\id}[1]{\renewcommand{\psu@id}{#1}}

% Store the title of the thesis.
\newcommand{\psu@title}{Theory of Everything}
\renewcommand{\title}[1]{\renewcommand{\psu@title}{#1}}

% Store the author's name.
\newcommand{\psu@author}{Richard Feynman}
\renewcommand{\author}[1]{\renewcommand{\psu@author}{#1}}

% Store the department name.
\newcommand{\psu@dept}{ESM}
\providecommand{\dept}[1]{\renewcommand{\psu@dept}{#1}}

% Store the date the degree will be conferred.
\newcommand{\psu@degreedate}{May 1900}
\providecommand{\degreedate}[1]{\renewcommand{\psu@degreedate}{#1}}

% Store the year of the copyright.
\newcommand{\psu@copyrightyear}{1900}
\providecommand{\copyrightyear}[1]{\renewcommand{\psu@copyrightyear}{#1}}

% Store the document type.
\newcommand{\psu@documenttype}{Thesis}
\providecommand{\documenttype}[1]{\renewcommand{\psu@documenttype}{#1}}

% Store the College to which the document has been submitted.
\newcommand{\psu@collegesubmittedto}{College of Engineering}
\providecommand{\collegesubmittedto}[1]{\renewcommand{\psu@collegesubmittedto}{#1}}

% Store the the info for the honors degree(s) type(s).
\newcommand{\psu@honorsdegreeinfo}{for a baccalaureate degree(s) \\ in Biology and Physics \\ with honors in Computer Engineering}
\providecommand{\honorsdegreeinfo}[1]{\renewcommand{\psu@honorsdegreeinfo}{#1}}

% Store the academic unit to which the document has been submitted.
\newcommand{\psu@honorsadviser}{Your Honors Adviser}
\providecommand{\honorsadviser}[1]{\renewcommand{\psu@honorsadviser}{#1}}

%Store the title of the academic advisor
\newcommand{\psu@honorsadvisertitle}{fill}
\providecommand{\honorsadvisertitle}[1]{\renewcommand{\psu@honorsadvisertitle}{#1}}

% Store the academic unit to which the document has been submitted.
\newcommand{\psu@honorsdepthead}{Department Q. Head}
\providecommand{\honorsdepthead}[1]{\renewcommand{\psu@honorsdepthead}{#1}}

% Store the name of the second Thesis Supervisor for a baccalaureate degree.
\newcommand{\psu@secondthesissupervisor}{Second Q. Jones}
\providecommand{\secondthesissupervisor}[1]{\renewcommand{\psu@secondthesissupervisor}{#1}}

%Store the title of the second thesis supervisor
\newcommand{\psu@secondthesissupervisortitle}{fill_two}
\providecommand{\secondthesissupervisortitle}[1]{\renewcommand{\psu@secondthesissupervisortitle}{#1}}


% Store the academic unit to which the document has been submitted.
\newcommand{\psu@submittedto}{The Graduate School}
\providecommand{\submittedto}[1]{\renewcommand{\psu@submittedto}{#1}}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Store the name of the degree by determining which boolean was
% set in the class option was specified.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\ifthenelse{\boolean{psu@bs}}%
{\newcommand{\psu@degreetype}{Baccalaureate of Science}}%
{}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Store the number of readers in \psu@readers. This quantity is
% input in the main file using the \numberofreaders command.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\psu@advisor}{John Doe}
\newcommand{\psu@advisortitle}{John Doe}
\newcommand{\psu@advisoroption}{}%
\providecommand{\advisor}[3][]%
{\renewcommand{\psu@advisoroption}{#1}%
\renewcommand{\psu@advisor}{#2}%
\renewcommand{\psu@advisortitle}{#3}}



\newsavebox{\tempbox}
\renewcommand{\@makecaption}[2]{%
\vspace{7pt}\sbox{\tempbox}{\small\textbf{#1.} #2}%
\ifthenelse{\lengthtest{\wd\tempbox > \linewidth}}%
{\small\textbf{#1.} #2\par}%
{\centering \small\textbf{#1.} #2}%
}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                %
% The actual layout begins here. %
%                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Here is the title page.
\newcommand{\psutitlepage}{%
%\setcounter{page}{1}
\thispagestyle{empty}%
\addtocounter{page}{-1}
%%%
\ifthenelse{\boolean{psu@honors}}{%if-TRUE
\vspace*{-0.75in}
%\enlargethispage{0.5in}
\begin{center}
THE PENNSYLVANIA STATE UNIVERSITY \\ SCHREYER HONORS COLLEGE
\end{center}
\vfill
\begin{center}
\MakeUppercase{\psu@dept}
\end{center}
\vfill
\begin{center}
\psu@title
\end{center}
\vfill
\begin{center}
\MakeUppercase{\psu@author} \\ \MakeUppercase{\psu@degreedate}
\end{center}
\vfill
\begin{singlespace}
\begin{center}
A thesis \\ submitted in partial fulfillment \\ of the requirements \\
\psu@honorsdegreeinfo
\end{center}
\vfill
%\mbox{}
\begin{center}
Reviewed and approved* by the following: \\*
\end{center}

%Names of Faculty Panel of Front Page:
\begin{center}
\psu@advisor \\ \psu@advisortitle \\ Thesis Supervisor \\* \vspace{0.1in}
\ifthenelse{\boolean{psu@honorsdepthead}}{%
\psu@honorsadviser \\ \psu@honorsadvisertitle \\ Honors Advisor\\* \vspace{0.1in}}%
{\psu@secondthesissupervisor \\ \psu@secondthesissupervisortitle \\ Faculty Reader \\* \vspace{0.1in}}
\psu@honorsdepthead \\ P. B. Breneman Department Head Chair \\ Professor, Department of Engineering Science and Mechanics
\end{center}
\end{singlespace}
\vfill

\setlength{\textwidth}{7.0in}
\parbox{\textwidth}{$^{*}$Signatures are on file in the Schreyer Honors College and The Engineering Science \\ and Mechanics Office}
\setlength{\textwidth}{5.8in}

%%%%%%%%%%%%Signature Page%%%%%%%%%%%%%%%%%%%%%
\ifthenelse{\boolean{psu@signature}}{
\newpage
\vspace*{-0.75in}
We approve the thesis of \psu@author \hfill

\thispagestyle{empty}  %Kill the Title
\addtocounter{page}{-1}
\vspace{0.5in}
\ifthenelse{\boolean{psu@honorsdepthead}}{%
\begin{tabbing}%
\= \rule{2.75in}{0.5pt} \quad Date: \= \rule{1.5in}{0.5pt} \\[-3pt]
          \> \qquad \psu@advisor \\[-3pt]
	\> \qquad \psu@advisortitle \\[-3pt]
          \> \qquad Thesis Supervisor \\[8mm]
%
\ifthenelse{\boolean{psu@secondthesissupervisor}}{%
          \> \rule{2.75in}{0.5pt}              \> \rule{1.5in}{0.5pt} \\[-3pt]
          \> \qquad \psu@secondthesissupervisor \\[-3pt]
	\> \qquad \psu@secondthesissupervisortitle \\[-3pt]
          \> \qquad Second Thesis Supervisor \\[8mm]
}{}%
%
          \> \rule{2.75in}{0.5pt}              \> \rule{1.5in}{0.5pt} \\[-3pt]
          \> \qquad \psu@honorsadviser \\[-3pt]
	\> \qquad \psu@honorsadvisertitle \\[-3pt]
          \> \qquad Honors Adviser \\[8mm]
 %
          \> \rule{2.75in}{0.5pt}              \> \rule{1.5in}{0.5pt} \\[-3pt]
          \> \qquad \psu@honorsdepthead \\[-3pt]
          \> \qquad Department Head
\end{tabbing}%

}%
{%
\begin{tabbing}%
\= \hspace{2.75in} \quad Date of Signature \\ \\[3pt]
 \= \rule{2.75in}{0.5pt} \quad  \= \rule{1.5in}{0.5pt} \\[-3pt]
          \> \qquad \psu@advisor \\[-3pt]
	\> \qquad \psu@advisortitle \\[-3pt]
          \> \qquad Thesis Supervisor \\[8mm]
	
%
\ifthenelse{\boolean{psu@secondthesissupervisor}}{%
          \> \rule{2.75in}{0.5pt}              \> \rule{1.5in}{0.5pt} \\[-3pt]
          \> \qquad \psu@secondthesissupervisor \\[-3pt]
	\> \qquad \psu@secondthesissupervisortitle \\[-3pt]
          \> \qquad Faculty Reader \\[8mm]
	
}{}%
%
          \> \rule{2.75in}{0.5pt}              \> \rule{1.5in}{0.5pt} \\[-3pt]
          \> \qquad \psu@honorsdepthead \\[-3pt]
          \> \qquad P. B. Breneman Department Head Chair \\ [-3pt]
	\> \qquad Professor, Department of Engineering Science and Mechanics \\ [8mm]
\end{tabbing}%
\vfill
\hfill \psu@id
}%
}{} %Close Signature
}{%if - FALSE aka option: bs
\vspace*{-0.75in}
%\enlargethispage{0.5in}
\begin{center}
THE PENNSYLVANIA STATE UNIVERSITY 
\end{center}
\vfill
\begin{center}
\MakeUppercase{\psu@dept}
\end{center}
\vfill
\begin{center}
\psu@title
\end{center}
\vfill
\begin{center}
\MakeUppercase{\psu@author} \\ \MakeUppercase{\psu@degreedate}
\end{center}
\vfill
\begin{singlespace}
\begin{center}
A thesis \\ submitted in partial fulfillment \\ of the requirements \\
\psu@honorsdegreeinfo
\end{center}
\vfill
%\mbox{}
\begin{center}
Reviewed and approved* by the following: \\*
\end{center}

%Names of Faculty Panel of Front Page:
\begin{center}
 \psu@advisor \\ \psu@advisortitle \\ Thesis Supervisor \\* \vspace{0.1in}
\ifthenelse{\boolean{psu@honorsdepthead}}{%
\psu@honorsadviser \\ \psu@honorsadvisertitle \\ Honors Advisor\\* \vspace{0.1in}}%
{\psu@secondthesissupervisor \\ \psu@secondthesissupervisortitle \\ Faculty Reader \\* \vspace{0.1in}}
\psu@honorsdepthead \\ P. B. Breneman Department Head Chair \\ Professor, Department of Engineering Science and Mechanics
\end{center}
\end{singlespace}
\vfill

\setlength{\textwidth}{7.0in}
\parbox{\textwidth}{$^{*}$Signatures are on file in The Engineering Science and Mechanics Office}
\setlength{\textwidth}{5.8in}

%%%%%%%%%%%%Signature Page%%%%%%%%%%%%%%%%%%%%%
\ifthenelse{\boolean{psu@signature}}{
\newpage
\vspace*{-0.75in}
We approve the thesis of \psu@author \hfill

\thispagestyle{empty}  %Kill the Title
\addtocounter{page}{-1}
\vspace{0.5in}
\ifthenelse{\boolean{psu@honorsdepthead}}{%
\begin{tabbing}%
\= \rule{2.75in}{0.5pt} \quad Date: \= \rule{1.5in}{0.5pt} \\[-3pt]
          \> \qquad \psu@advisor \\[-3pt]
	\> \qquad \psu@advisortitle \\[-3pt]
          \> \qquad Thesis Supervisor \\[8mm]
%
\ifthenelse{\boolean{psu@secondthesissupervisor}}{%
          \> \rule{2.75in}{0.5pt}              \> \rule{1.5in}{0.5pt} \\[-3pt]
          \> \qquad \psu@secondthesissupervisor \\[-3pt]
	\> \qquad \psu@secondthesissupervisortitle \\[-3pt]
          \> \qquad Second Thesis Supervisor \\[8mm]
}{}%
%
          \> \rule{2.75in}{0.5pt}              \> \rule{1.5in}{0.5pt} \\[-3pt]
          \> \qquad \psu@honorsadviser \\[-3pt]
	\> \qquad \psu@honorsadvisertitle \\[-3pt]
          \> \qquad Honors Adviser \\[8mm]
 %
          \> \rule{2.75in}{0.5pt}              \> \rule{1.5in}{0.5pt} \\[-3pt]
          \> \qquad \psu@honorsdepthead \\[-3pt]
          \> \qquad Department Head
\end{tabbing}%

}%
{%
\begin{tabbing}%
\= \hspace{2.75in} \quad Date of Signature \\ \\[3pt]
 \= \rule{2.75in}{0.5pt} \quad  \= \rule{1.5in}{0.5pt} \\[-3pt]
          \> \qquad \psu@advisor \\[-3pt]
	\> \qquad \psu@advisortitle \\[-3pt]
          \> \qquad Thesis Supervisor \\[8mm]
	
%
\ifthenelse{\boolean{psu@secondthesissupervisor}}{%
          \> \rule{2.75in}{0.5pt}              \> \rule{1.5in}{0.5pt} \\[-3pt]
          \> \qquad \psu@secondthesissupervisor \\[-3pt]
	\> \qquad \psu@secondthesissupervisortitle \\[-3pt]
          \> \qquad Faculty Reader \\[8mm]
	
}{}%
%
          \> \rule{2.75in}{0.5pt}              \> \rule{1.5in}{0.5pt} \\[-3pt]
          \> \qquad \psu@honorsdepthead \\[-3pt]
          \> \qquad P. B. Breneman Department Head Chair \\ [-3pt]
	\> \qquad Professor, Department of Engineering Science and Mechanics \\ [8mm]
\end{tabbing}%
}%
\vfill
\hfill \psu@id
}{} %Close Signature
}%
%%%
}%Close honors option


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Abstract
\setcounter{page}{1}
\newcommand{\thesisabstract}[1]{%
\pagestyle{plain}
\chapter*{Abstract}
    \begin{singlespace}
        \input{#1}
    \end{singlespace}
\newpage
}

\newcommand{\thesisnontechabstract}[1]{%
\pagestyle{plain}
\chapter*{Non-Technical Abstract}
    \begin{singlespace}
        \input{#1}
    \end{singlespace}
\newpage
}


\renewcommand{\contentsname}{Table of Contents}
\setcounter{secnumdepth}{10}
\setcounter{tocdepth}{4}
\newcommand{\thesistableofcontents}{%
\begin{singlespace}
\if@twocolumn
\@restonecoltrue\onecolumn
\else
\@restonecolfalse
\fi
\chapter*{\contentsname
\@mkboth{%
\MakeUppercase\contentsname}{\MakeUppercase\contentsname}}%
\@starttoc{toc}%
\if@restonecol\twocolumn\fi%
\end{singlespace}
\clearpage
}

\renewcommand{\listfigurename}{List of Figures}
\newcommand{\thesislistoffigures}{%
\begin{singlespace}
\if@twocolumn
\@restonecoltrue\onecolumn
\else
\@restonecolfalse
\fi
\chapter{\listfigurename
\@mkboth{%
\MakeUppercase\listfigurename}{\MakeUppercase\listfigurename}}%
\@starttoc{lof}%
\if@restonecol\twocolumn\fi
\end{singlespace}
\clearpage
}

\renewcommand{\listtablename}{List of Tables}
\newcommand{\thesislistoftables}{%
\begin{singlespace}
\if@twocolumn
\@restonecoltrue\onecolumn
\else
\@restonecolfalse
\fi
\chapter{\listtablename
\@mkboth{%
\MakeUppercase\listtablename}{\MakeUppercase\listtablename}}%
\@starttoc{lot}%
\if@restonecol\twocolumn\fi
\end{singlespace}
\clearpage
}


\newcommand{\thesislistofsymbols}[1]{%
\chapter{List of Symbols}
%\addcontentsline{toc}{chapter}{List of Symbols}
\begin{singlespace}
    \input{#1}
\end{singlespace}
}



\newcommand{\thesisacknowledgments}[1]{%
\chapter{Acknowledgments}
%\addcontentsline{toc}{chapter}{Acknowledgments}
\begin{singlespace}
    \input{#1}
\end{singlespace}
}


\newcommand{\thesisdedication}[2]{%
\chapter*{#2}
\begin{singlespace}
    \input{#1}
\end{singlespace}
}




\newcommand{\Appendix}[1]{%
\ifthenelse{\value{chapter} = 0}
{
\ifthenelse{\boolean{psu@toc}}
{%
\addtocontents{toc}{\protect\addtolength{\cftchapnumwidth}{-\mylength}}
\addtocontents{toc}{\string\renewcommand{\string\cftchappresnum}{Appendix }}
\addtocontents{toc}{\protect\addtolength{\cftchapnumwidth}{\mylonglength}}
}%
{%
\addtocontents{toc}{\string\renewcommand{\string\cftchappresnum}{Appendix }}
}}{}%
\chapter{#1}
}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\DeclareRobustCommand{\thesismainmatter}{%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\setlength{\textheight}{8.5in}
\mainmatter
%\pagestyle{empty}
%\renewcommand{\@oddhead}{\mbox{}\hfill\arabic{page}}
%\let\ps@plain = \ps@empty
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\vita}[1]{%
\begin{singlespace}
\thispagestyle{empty}
\begin{center}
    \textbf{{\large Vita} \\[0.1in] \psu@author}
\end{center}
\input{#1}
\end{singlespace}
}


%%
%% End of file `psuthesis.cls'.
