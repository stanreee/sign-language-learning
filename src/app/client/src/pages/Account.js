import { useNavigate } from 'react-router-dom';
import asl from '../images/asl.png';

console.log(asl);

const Account = (props) => {

  const { loggedIn, email, name } = props
  const navigate = useNavigate();
  
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

      <div className="mainContainer">
        <div className={'titleContainer'}>
          {loggedIn ? <div>Welcome {name}!</div> : <div>Welcome to ASLingo</div>}
          {/* name part isnt working right now */}
        </div>
        {loggedIn ? <h2>Profile Settings</h2> : <div />}
        {loggedIn ? <div>You are signed in with {email}</div> : <div>Login or Create an Account to Save Your Learning Progress!</div>}

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