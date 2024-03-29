import { useNavigate } from 'react-router-dom';
import asl from '../images/asl.png';

console.log(asl);

const Account = (props) => {

  const { loggedIn, userId, email, name, level } = props
  const navigate = useNavigate();

  console.log(props)
  
  const onLoginButtonClick = () => {
    if (loggedIn) {
        localStorage.removeItem("user")
        props.setLoggedIn(false)
    } else {
        navigate("/login")
    }
  }

  const onSignUpButtonClick = () => {
    if (loggedIn) {
        localStorage.removeItem("user")
        props.setLoggedIn(false)
    } else {
        navigate("/signup")
    }
  }

    return(
      <div className='container'>
        <div className="h2" style={{textAlign: "left"}}> Profile Settings </div>

        <br />
        <br />
        <br />
        <br />
        <br />
        <br />
        <br />

        <div className='titleContainer'>
          {/* name doesn't work?? */}
          {loggedIn ? 
          <div style={{textAlign: "center"}}>
            Welcome {name}!
            <div className="h2">You are signed in with {email}</div>
          </div> : <div>Welcome to ASLingo</div>}
        </div>

        <br />
        {loggedIn ? <div className="h2" style={{textAlign: "center"}}> Learning Statistics: </div> : <div />}
        {loggedIn ? 
          <div className="box">
            <div className="h2">Current Level: {level[2]}</div>
            <br />
            <div className="h2">Overall Exercise Score: {level[0]}/{level[1]} total questions</div>
          </div>
          
          : <div className="h2" style={{textAlign: "center"}}>
              Login or Create an Account to Save Your Learning Progress!
            </div>
        }

        <div className={'buttonContainer'}>
          <input
            className={'inputButton'}
            type="button"
            onClick={onLoginButtonClick}
            value={loggedIn ? 'Log out' : 'Log in'}
          />

          <input
            className={'inputButton'}
            type={loggedIn ? 'hidden' : 'button'}
            onClick={onSignUpButtonClick}
            value={loggedIn ? '' : 'Sign Up'}
          />
        </div>
      </div>

      );
}

export default Account;