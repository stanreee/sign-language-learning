import a from "../../images/learnLetters/a.png"
import b from "../../images/learnLetters/b.png"
import c from "../../images/learnLetters/c.png"
import d from "../../images/learnLetters/d.png"
import e from "../../images/learnLetters/e.png"
import f from "../../images/learnLetters/f.png"
import g from "../../images/learnLetters/g.png"
import h from "../../images/learnLetters/h.png"
import i from "../../images/learnLetters/i.png"
import j from "../../images/learnLetters/j.png"
import k from "../../images/learnLetters/k.png"
import l from "../../images/learnLetters/l.png"
import m from "../../images/learnLetters/m.png"
import n from "../../images/learnLetters/n.png"
import o from "../../images/learnLetters/o.png"
import p from "../../images/learnLetters/p.png"
import q from "../../images/learnLetters/q.png"
import r from "../../images/learnLetters/r.png"
import s from "../../images/learnLetters/s.png"
import t from "../../images/learnLetters/t.png"
import u from "../../images/learnLetters/u.png"
import v from "../../images/learnLetters/v.png"
import w from "../../images/learnLetters/w.png"
import x from "../../images/learnLetters/x.png"
import y from "../../images/learnLetters/y.png"
import z from "../../images/learnLetters/z.png"
import youtube from "../../images/learnLetters/youtube.png"

import lettersResource from "../../data/ASLSigns.json"

import { Link } from "react-router-dom"

const letterCard = (letter: string, num: number, first?: boolean) => {
    // DATA CALL
    const allLetters = lettersResource.allLetters;
    const letterDesc = lettersResource.lettersDesc;
    const letterVideos = lettersResource.letterVideos;

    return (
      <div className={first ? "card first" : "card"}>
        <img src={letter} alt= {`${allLetters[num]}`} />
        <h2>{allLetters[num][0]}</h2>
        {letterDesc[num]}
        <Link 
          to={`https://youtu.be/${letterVideos[num]}`}
          style={{color: '#000000',}}
          > <div > <img src={youtube} width={46.2} height={32.4} alt= "youtube" /> </div>
        </Link>
      </div>
    )
}

const Letters = (props: any) => {
    const {
        className
    } = props;

    return (
        <div style={{overflow: "hidden", width: "75%"}} className={className}>
            <div className="container-letters">
                <div id="A" data-hs-anchor="true">{letterCard(a, 0, true)}</div>   
                <div id="B" data-hs-anchor="true">{letterCard(b, 1)}</div> 
                <div id="C" data-hs-anchor="true">{letterCard(c, 2)}</div>
                <div id="D" data-hs-anchor="true">{letterCard(d, 3)}</div> 
                <div id="E" data-hs-anchor="true">{letterCard(e, 4)}</div>
                <div id="F" data-hs-anchor="true">{letterCard(f, 5)}</div> 
                <div id="G" data-hs-anchor="true">{letterCard(g, 6)}</div>
                <div id="H" data-hs-anchor="true">{letterCard(h, 7)}</div> 
                <div id="I" data-hs-anchor="true">{letterCard(i, 8)}</div>
                <div id="J" data-hs-anchor="true">{letterCard(j, 9)}</div>  
                <div id="K" data-hs-anchor="true">{letterCard(k, 10)}</div> 
                <div id="L" data-hs-anchor="true">{letterCard(l, 11)}</div> 
                <div id="M" data-hs-anchor="true">{letterCard(m, 12)}</div> 
                <div id="N" data-hs-anchor="true">{letterCard(n, 13)}</div>
                <div id="O" data-hs-anchor="true">{letterCard(o, 14)}</div> 
                <div id="P" data-hs-anchor="true">{letterCard(p, 15)}</div> 
                <div id="Q" data-hs-anchor="true">{letterCard(q, 16)}</div>
                <div id="R" data-hs-anchor="true">{letterCard(r, 17)}</div>  
                <div id="S" data-hs-anchor="true">{letterCard(s, 18)}</div>
                <div id="T" data-hs-anchor="true">{letterCard(t, 19)}</div> 
                <div id="U" data-hs-anchor="true">{letterCard(u, 20)}</div>  
                <div id="V" data-hs-anchor="true">{letterCard(v, 21)}</div> 
                <div id="W" data-hs-anchor="true">{letterCard(w, 22)}</div>  
                <div id="X" data-hs-anchor="true">{letterCard(x, 23)}</div>  
                <div id="Y" data-hs-anchor="true">{letterCard(y, 24)}</div> 
                <div id="Z" data-hs-anchor="true">{letterCard(z, 25)}</div> 
            </div>
            <br />
            <div className="jump-to">
                Jump to: 
                <br />
                <a href="#A" rel="noopener">A   </a>
                <a href="#B" rel="noopener">B   </a>
                <a href="#C" rel="noopener">C   </a>
                <a href="#D" rel="noopener">D   </a>
                <a href="#E" rel="noopener">E   </a>
                <a href="#F" rel="noopener">F   </a>
                <a href="#G" rel="noopener">G   </a>
                <a href="#H" rel="noopener">H   </a>
                <a href="#I" rel="noopener">I   </a>
                <a href="#J" rel="noopener">J   </a>
                <a href="#K" rel="noopener">K   </a>
                <a href="#L" rel="noopener">L   </a>
                <a href="#M" rel="noopener">M   </a>
                <a href="#N" rel="noopener">N   </a>
                <a href="#O" rel="noopener">O   </a>
                <a href="#P" rel="noopener">P   </a>
                <a href="#Q" rel="noopener">Q   </a>
                <a href="#R" rel="noopener">R   </a>
                <a href="#S" rel="noopener">S   </a>
                <a href="#T" rel="noopener">T   </a>
                <a href="#U" rel="noopener">U   </a>
                <a href="#V" rel="noopener">V   </a>
                <a href="#W" rel="noopener">W   </a>
                <a href="#X" rel="noopener">X   </a>
                <a href="#Y" rel="noopener">Y   </a>
                <a href="#Z" rel="noopener">Z   </a>
            </div>
        </div>
    );
}

export default Letters;